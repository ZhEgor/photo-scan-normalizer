"""Detect and extract individual photos from a flatbed scan sheet.

A scan usually has a white background, so photos are the non-white regions.
Each region is isolated, deskewed (small rotations are corrected), rotated
upright (sideways and upside-down prints are detected from their content),
cropped to a tight rectangle and written out as a separate image.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from .orientation import Orientation, apply_rotation, estimate_orientations

__all__ = [
    "DetectedPhoto",
    "discover_images",
    "load_image",
    "normalize_image",
    "save_image",
    "split_file",
    "split_image",
]

IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".ppm", ".pgm"}
)

_WHITE_BGR = (255, 255, 255)

# Real photos are never long thin strips, but scanner lid edges and shadows are.
_MAX_ASPECT = 20.0
_MIN_INK_DENSITY = 0.05

# A candidate that is barely filled (a sparse scanner streak, not a print) is
# only used to extend a neighbouring photo, never reported on its own.
_MIN_FILL = 0.25

# Faded black-and-white prints can be white-on-white: their content is a few
# disconnected shadows on a sheet that looks like the scanner bed. The paper
# itself, however, keeps a little texture, so the paper threshold treats
# everything that is not pure scanner white as foreground.
_PAPER_LEVEL = 254
_PAPER_FALLBACK_COVERAGE = 0.9  # background is not pure white: use --threshold

# Cleaning kernel, as a fraction of the shorter sheet side. Removes dust and
# scanner streaks that are thinner than this without eroding the photos much.
_OPEN_RATIO = 0.0035

# A thin strip hugging the sheet border is a scanner edge/shadow, not a print.
_EDGE_STRIP_RATIO = 0.12

# A bright band spanning a photo-like region is the gap between two prints
# (the scanner bed in shadow), not part of a print.
_SEAM_LEVEL = 235
_SEAM_MAX_RATIO = 0.25
_SEAM_DARK_LEVEL = 200
_SEAM_DARK_FRACTION = 0.25

# Merging the pieces of one print is only trusted when the paper region holds
# no second photo-sized piece: a print can be mostly white, but a region that
# spans two real prints shows two of them.
_FRAGMENT_DOMINANCE = 0.25
_PAPER_REGION_COVERAGE = 0.9  # refuse to trust a "paper" region covering it all


@dataclass(frozen=True)
class DetectedPhoto:
    """One cropped photo and where it came from."""

    image: np.ndarray
    box: tuple[int, int, int, int]  # x, y, w, h in the source image
    angle: float  # deskew angle applied, degrees
    corners: tuple[tuple[float, float], ...]  # source-space photo corners
    rotation: int = 0  # uprighting rotation applied, clockwise degrees
    rotation_confidence: float = 0.0
    rotation_basis: str = "none"  # faces | sheet | colour | none


def load_image(path: str | Path) -> np.ndarray:
    """Read an image from disk, tolerating non-ASCII paths."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"cannot read image (empty file): {path}")
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return normalize_image(image)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit BGR image, compositing any alpha over white."""
    if image.dtype == np.uint16:
        image = (image.astype(np.uint32) * 255 // 65535).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3:
        raise ValueError(f"unsupported image shape: {image.shape}")

    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 3:
        return np.ascontiguousarray(image)
    if channels == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        color = image[:, :, :3].astype(np.float32)
        out = color * alpha + 255.0 * (1.0 - alpha)
        return out.round().astype(np.uint8)
    raise ValueError(f"unsupported channel count: {channels}")


def save_image(image: np.ndarray, path: str | Path, *, quality: int = 95) -> Path:
    """Write an image, tolerating non-ASCII paths."""
    path = Path(path)
    suffix = path.suffix.lower()
    params: list[int] = []
    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 1, 100))]
    elif suffix == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(np.clip(quality, 1, 100))]

    ok, buffer = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"cannot encode image as {suffix or 'unknown format'}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(str(path))
    return path


def discover_images(paths: list[str | Path], *, recursive: bool = False) -> list[Path]:
    """Expand files and directories into a sorted list of image paths."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            found.extend(
                p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")
    return sorted(dict.fromkeys(found))


def split_file(
    src: str | Path,
    *,
    white_threshold: int = 240,
    min_area_ratio: float = 0.002,
    padding: int = 0,
    deskew: bool = True,
    trim: bool = True,
    trim_threshold: int | None = None,
    rotate: int | str = "auto",
) -> list[DetectedPhoto]:
    """Load ``src`` and split it into individual photos."""
    return split_image(
        load_image(src),
        white_threshold=white_threshold,
        min_area_ratio=min_area_ratio,
        padding=padding,
        deskew=deskew,
        trim=trim,
        trim_threshold=trim_threshold,
        rotate=rotate,
    )


def split_image(
    image: np.ndarray,
    *,
    white_threshold: int = 240,
    min_area_ratio: float = 0.002,
    padding: int = 0,
    deskew: bool = True,
    trim: bool = True,
    trim_threshold: int | None = None,
    rotate: int | str = "auto",
) -> list[DetectedPhoto]:
    """Split a scan on a white background into separate photo images.

    Args:
        image: BGR, BGRA or grayscale source image.
        white_threshold: Pixels whose darkest channel is >= this value are
            treated as background. Lower it for greyish paper, raise it for
            off-white backgrounds.
        min_area_ratio: Ignore blobs smaller than this fraction of the sheet.
        padding: White pixels to add around every extracted photo.
        deskew: Straighten small rotations.
        trim: Remove leftover uniform bright bands (scanner shadow, antialiased
            edges) from the border of every photo.
        trim_threshold: Brightness at which a border line counts as background.
            Defaults to ``white_threshold - 10``.
        rotate: ``"auto"`` (default) detects sideways and upside-down prints
            from their content, ``"none"``/``0`` keeps the scanned orientation,
            and 90/180/270 forces one rotation on every photo.
    """
    bgr = normalize_image(image)
    if bgr.size == 0:
        return []

    if trim_threshold is None:
        trim_threshold = max(0, white_threshold - 10)

    mask = _photo_mask(bgr, white_threshold)
    rects = _find_photo_rectangles(bgr, mask, min_area_ratio)

    photos: list[DetectedPhoto] = []
    for rect in rects:
        center, size, angle, corners = _deskew_geometry(rect)
        photo, box = _extract_photo(
            bgr,
            center=center,
            size=size,
            angle=angle,
            corners=corners,
            padding=max(0, int(padding)),
            deskew=deskew,
            trim=trim,
            trim_threshold=trim_threshold,
        )
        photos.append(
            DetectedPhoto(
                image=photo,
                box=box,
                angle=angle if deskew else 0.0,
                corners=corners,
            )
        )
    return _apply_orientation(bgr, _reading_order(photos), rotate)


def _apply_orientation(
    sheet: np.ndarray, photos: list[DetectedPhoto], rotate: int | str
) -> list[DetectedPhoto]:
    """Rotate each photo upright, either forced or from detected content."""
    if not photos or rotate is None:
        return photos

    if isinstance(rotate, str) and rotate.lower() == "none":
        return photos

    if isinstance(rotate, str) and rotate.lower() == "auto":
        estimates = estimate_orientations(sheet, photos)
    else:
        angle = int(rotate) % 360
        if angle not in (0, 90, 180, 270):
            raise ValueError(f"rotate must be auto, none, 0, 90, 180 or 270: {rotate}")
        estimates = [
            Orientation(rotation=angle, confidence=1.0, basis="forced")
            for _ in photos
        ]

    return [
        replace(
            photo,
            image=apply_rotation(photo.image, estimate.rotation),
            rotation=estimate.rotation,
            rotation_confidence=estimate.confidence,
            rotation_basis=estimate.basis,
        )
        for photo, estimate in zip(photos, estimates)
    ]


def _foreground_mask(bgr: np.ndarray, white_threshold: int) -> np.ndarray:
    """255 where the sheet is not white."""
    darkest = bgr.min(axis=2)
    return np.where(darkest < white_threshold, 255, 0).astype(np.uint8)


def _photo_mask(bgr: np.ndarray, white_threshold: int) -> np.ndarray:
    """Foreground for photo detection, falling back to ``white_threshold``.

    Scanner beds are clipped to pure white, while the paper of a faded print is
    a little below it. Treating "not pure white" as foreground keeps the whole
    print together, white margins included. On a sheet whose background is not
    clipped white that would select everything, so the requested threshold is
    used instead.
    """
    height, width = bgr.shape[:2]
    strong = _foreground_mask(bgr, white_threshold)
    paper = np.where(bgr.min(axis=2) < _PAPER_LEVEL, 255, 0).astype(np.uint8)
    if cv2.countNonZero(paper) > _PAPER_FALLBACK_COVERAGE * height * width:
        return strong
    return paper


def _odd_kernel(value: float) -> int:
    size = max(3, int(round(value)))
    return size + 1 if size % 2 == 0 else size


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Drop dust and hairline scanner streaks thinner than the kernel."""
    height, width = mask.shape[:2]
    size = _odd_kernel(min(height, width) * _OPEN_RATIO)
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
    )


def _find_photo_rectangles(
    bgr: np.ndarray, mask: np.ndarray, min_area_ratio: float
) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    """Find one rotated rectangle per photo-like blob, reading order unsorted."""
    height, width = mask.shape[:2]
    total = float(height * width)
    if total <= 0:
        return []

    min_area = max(float(min_area_ratio) * total, 64.0)
    cleaned = _clean_mask(mask)

    # A scanner edge/lid shadow often forms a frame around the whole sheet.
    # Photos inside it get merged into one huge blob, so strip the frame and
    # re-detect when that happens.
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if any(_covers_sheet(cv2.minAreaRect(c), width, height) for c in contours):
        ring = _edge_ring(cleaned)
        max_ring = max(3, int(min(height, width) * 0.02))
        if 2 <= ring <= max_ring:
            cleaned[:ring, :] = 0
            cleaned[-ring:, :] = 0
            cleaned[:, :ring] = 0
            cleaned[:, -ring:] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    gray = bgr.min(axis=2)
    main: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    aux: list[tuple[int, int, int, int]] = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < min_area:
            continue
        component = np.where(labels == index, 255, 0).astype(np.uint8)
        component = _trim_appendages(component)
        pieces = []
        for part in _mask_pieces(component, min_area):
            pieces.extend(_split_seams(part, gray, min_area))
        for piece in pieces:
            piece = _trim_appendages(piece)
            for sub in _mask_pieces(piece, min_area):
                rect = _piece_rectangle(sub, cleaned, min_area)
                if rect is None:
                    continue
                box = cv2.boundingRect(cv2.boxPoints(rect).astype(np.int32))
                short_side, long_side = sorted(rect[1])
                fill = cv2.countNonZero(sub) / max(1.0, _area(box))
                if long_side / max(1.0, short_side) > _MAX_ASPECT or fill < _MIN_FILL:
                    aux.append(box)  # sparse streak or a photo's edge line
                    continue
                if _edge_strip(box, width, height):
                    continue
                main.append(rect)

    original: dict[tuple[int, int, int, int], tuple] = {}
    for rect in main:
        original.setdefault(
            cv2.boundingRect(cv2.boxPoints(rect).astype(np.int32)), rect
        )

    regions = _paper_regions(bgr, cleaned)
    boxes = list(original)
    boxes = _drop_outside(boxes, width, height)
    boxes = _drop_nested(boxes)
    boxes = _merge_fragments(bgr, boxes, regions=regions)
    boxes = _merge_boxes(boxes, aux, width, height, min_area)
    boxes = _merge_fragments(bgr, boxes, regions=regions)

    refined = []
    for box in boxes:
        rect = original.get(box)
        if rect is not None:
            refined.append(rect)
            continue
        x, y, w, h = box
        refined.append(((x + w / 2.0, y + h / 2.0), (float(w), float(h)), 0.0))
    return refined


def _mask_pieces(mask: np.ndarray, min_area: float) -> list[np.ndarray]:
    """Split a component after trimming; usually it stays a single piece."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    pieces = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < min_area:
            continue
        pieces.append(np.where(labels == index, 255, 0).astype(np.uint8))
    return pieces


def _split_seams(
    mask: np.ndarray, gray: np.ndarray, min_area: float, depth: int = 0
) -> list[np.ndarray]:
    """Separate prints joined by a bright seam (scanner bed in shadow).

    The seam is not pure white, so the paper mask includes it; but it is much
    brighter than print content and spans the whole region. Splitting there
    keeps two prints that almost touch apart.
    """
    if depth > 4:
        return [mask]
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return [mask]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    window = mask[y0:y1, x0:x1] > 0
    gray_window = gray[y0:y1, x0:x1].astype(np.float64)

    for direction in ("rows", "cols"):
        axis = 1 if direction == "rows" else 0
        counts = window.sum(axis=axis)
        sums = (gray_window * window).sum(axis=axis)
        means = sums / np.maximum(1, counts)
        seam = means >= _SEAM_LEVEL
        length = len(seam)
        context = max(3, int(round(0.02 * length)))
        position = 0
        while position < length:
            if not seam[position]:
                position += 1
                continue
            start = position
            while position < length and seam[position]:
                position += 1
            end = position
            if start <= 1 or end >= length - 1:
                continue
            if end - start > _SEAM_MAX_RATIO * length:
                continue
            if end - start < max(4, int(round(0.005 * length))):
                continue
            left = slice(max(0, start - context), start)
            right = slice(end, min(length, end + context))
            left_valid = counts[left] > 0
            right_valid = counts[right] > 0
            if not (left_valid.any() and right_valid.any()):
                continue
            seam_mean = float(means[start:end].mean())
            left_mean = float(means[left][left_valid].mean())
            right_mean = float(means[right][right_valid].mean())
            # Paper does not border its own margin with more paper: both
            # sides of a real seam are darker print content.
            if left_mean > seam_mean - 8 or right_mean > seam_mean - 8:
                continue
            cut = mask.copy()
            if direction == "rows":
                cut[y0 + start : y0 + end, :] = 0
            else:
                cut[:, x0 + start : x0 + end] = 0
            pieces = _mask_pieces(cut, min_area)
            if len(pieces) < 2:
                continue
            # A print's own white margin splits off a sliver; a real seam
            # leaves two substantial printed areas behind.
            total_area = sum(cv2.countNonZero(piece) for piece in pieces)
            smallest = min(cv2.countNonZero(piece) for piece in pieces)
            if smallest < 0.25 * total_area:
                continue
            if (
                min(_dark_fraction(piece, gray) for piece in pieces)
                < _SEAM_DARK_FRACTION
            ):
                continue
            result = []
            for piece in pieces:
                result.extend(_split_seams(piece, gray, min_area, depth + 1))
            return result
    return [mask]


def _dark_fraction(piece: np.ndarray, gray: np.ndarray) -> float:
    """Fraction of a piece that is clearly darker than paper/scanner white."""
    values = gray[piece > 0]
    return float((values < _SEAM_DARK_LEVEL).mean()) if values.size else 0.0


def _piece_rectangle(
    piece: np.ndarray,
    mask: np.ndarray,
    min_area: float,
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    """Fit a rectangle to one candidate piece, or reject it."""
    contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    rect = cv2.minAreaRect(contour)
    short_side, _ = sorted(rect[1])
    if short_side < 8 or rect[1][0] * rect[1][1] < min_area:
        return None
    if _ink_density(mask, rect) < _MIN_INK_DENSITY:
        return None
    return rect


def _trim_appendages(mask: np.ndarray) -> np.ndarray:
    """Cut flat, narrow strips hanging off a print (scanner edge shadows).

    A print is a filled rectangle, so its width/height profile is flat. A
    scanner shadow attached to its border shows up as a long, narrow band at
    one end of that profile. Rounded corners of a slightly tilted print are a
    ramp instead of a flat band, so they are left alone.
    """
    out = mask
    for direction in ("rows", "cols"):
        ys, xs = np.nonzero(out)
        if ys.size == 0:
            return mask
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub = out[y0:y1, x0:x1]
        axis = 1 if direction == "rows" else 0
        profile = (sub > 0).sum(axis=axis).astype(np.float64)
        peak = profile.max() if profile.size else 0.0
        if peak <= 0:
            continue
        wide = np.flatnonzero(profile >= 0.85 * peak)
        if wide.size == 0:
            continue
        low, high = int(wide[0]), int(wide[-1])
        if high - low + 1 < 0.3 * len(profile):
            continue
        for band, leading in ((profile[:low], True), (profile[high + 1 :], False)):
            if band.size == 0:
                continue
            if band.mean() >= 0.5 * peak or band.std() > 0.15 * peak:
                continue
            if direction == "rows":
                if leading:
                    out[: y0 + low, :] = 0
                else:
                    out[y0 + high + 1 :, :] = 0
            else:
                if leading:
                    out[:, : x0 + low] = 0
                else:
                    out[:, x0 + high + 1 :] = 0
    return out


def _edge_strip(
    box: tuple[int, int, int, int], width: int, height: int
) -> bool:
    """True for a narrow band hugging the sheet border: scanner shadow."""
    x, y, w, h = box
    margin_x = max(4, int(width * 0.02))
    margin_y = max(4, int(height * 0.02))
    aspect = max(w, h) / max(1.0, min(w, h))
    if aspect < 5.0:
        return False
    if (
        x <= margin_x or x + w >= width - margin_x
    ) and w <= _EDGE_STRIP_RATIO * width:
        return True
    if (
        y <= margin_y or y + h >= height - margin_y
    ) and h <= _EDGE_STRIP_RATIO * height:
        return True
    return False


def _edge_gap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int]:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    return (
        max(0, max(bx0 - (ax0 + aw), ax0 - (bx0 + bw))),
        max(0, max(by0 - (ay0 + ah), ay0 - (by0 + bh))),
    )


def _is_whole_print(
    box: tuple[int, int, int, int], width: int, height: int
) -> bool:
    return box[2] >= 0.25 * width and box[3] >= 0.25 * height


def _same_print(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    width: int,
    height: int,
) -> bool:
    """True when two boxes are pieces of one print, not two separate prints."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    inter = _intersection(a, b)
    small = min(_area(a), _area(b))
    large = max(_area(a), _area(b))
    if small <= 0:
        return False
    if small >= 0.6 * large:
        return inter >= 0.6 * small
    tiny = a if _area(a) <= _area(b) else b
    if _is_whole_print(tiny, width, height):
        return inter >= 0.6 * small
    x_overlap = max(0, min(ax1, bx1) - max(ax0, bx0))
    y_overlap = max(0, min(ay1, by1) - max(ay0, by0))
    hgap, vgap = _edge_gap(a, b)
    if x_overlap >= 0.7 * min(aw, bw) and vgap <= 0.25 * min(aw, bw):
        uw = max(ax1, bx1) - min(ax0, bx0)
        uh = max(ay1, by1) - min(ay0, by0)
        if uh <= 3.0 * uw:
            return True
    if y_overlap >= 0.7 * min(ah, bh) and hgap <= 0.25 * min(ah, bh):
        uw = max(ax1, bx1) - min(ax0, bx0)
        uh = max(ay1, by1) - min(ay0, by0)
        if uw <= 3.0 * uh:
            return True
    if inter >= 0.15 * small:
        return True
    return False


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]],
    aux: list[tuple[int, int, int, int]],
    width: int,
    height: int,
    min_area: float,
) -> list[tuple[int, int, int, int]]:
    """Attach edge strips, then reunite aligned fragments of one print."""
    groups = [list(box) for box in boxes]

    margin = max(4.0, min(width, height) * 0.02)
    for ax, ay, aw, ah in aux:
        best = None
        for group in groups:
            gx, gy, gw, gh = group
            hgap, vgap = _edge_gap((ax, ay, aw, ah), (gx, gy, gw, gh))
            if hgap > margin or vgap > margin:
                continue
            overlap = max(
                max(0, min(ax + aw, gx + gw) - max(ax, gx)),
                max(0, min(ay + ah, gy + gh) - max(ay, gy)),
            )
            if overlap >= 0.3 * min(min(aw, gw), min(ah, gh)):
                best = group
                break
        if best is not None:
            x0 = min(best[0], ax)
            y0 = min(best[1], ay)
            x1 = max(best[0] + best[2], ax + aw)
            y1 = max(best[1] + best[3], ay + ah)
            best[:] = [x0, y0, x1 - x0, y1 - y0]

    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                if _same_print(tuple(a), tuple(b), width, height):
                    x0 = min(a[0], b[0])
                    y0 = min(a[1], b[1])
                    x1 = max(a[0] + a[2], b[0] + b[2])
                    y1 = max(a[1] + a[3], b[1] + b[3])
                    groups[i] = [x0, y0, x1 - x0, y1 - y0]
                    del groups[j]
                    changed = True
                    break
            if changed:
                break

    kept = []
    for group in groups:
        box = tuple(group)
        if _area(box) < min_area:
            continue
        if max(box[2], box[3]) / max(1, min(box[2], box[3])) > _MAX_ASPECT:
            continue
        kept.append(box)
    return kept


def _drop_outside(
    boxes: list[tuple[int, int, int, int]], width: int, height: int
) -> list[tuple[int, int, int, int]]:
    """Discard detections that reach well beyond the sheet: always artifacts.

    A rotated rectangle fitted to a shadow along the scanner border can point
    outside the image, which no real print ever does.
    """
    margin_x = max(4.0, width * 0.02)
    margin_y = max(4.0, height * 0.02)
    kept = []
    for x, y, w, h in boxes:
        if w < 8 or h < 8:
            continue
        if x < -margin_x or y < -margin_y:
            continue
        if x + w > width + margin_x or y + h > height + margin_y:
            continue
        kept.append((x, y, w, h))
    return kept


def _drop_nested(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Keep the outermost of overlapping detections.

    A bright print with a dark inner frame can be found twice; the inner copy
    is a duplicate of the same photo, not a second one.
    """
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=_area, reverse=True):
        if any(_intersection(box, other) >= 0.9 * _area(box) for other in kept):
            continue
        kept.append(box)
    return kept


def _merge_fragments(
    bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    *,
    regions: list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Reunite the pieces of one faded print into a single photo.

    Only done when the paper of the print is visible around the pieces and
    nothing separates them: a paper region barely larger than its biggest
    piece, no second full-sized piece inside it, and no seam line between
    them. Two prints lying side by side fail at least one of those.
    """
    if len(boxes) < 2:
        return boxes

    if regions is None:
        regions = _paper_regions(bgr)
    if not regions:
        return boxes

    boxes = list(boxes)
    for region in sorted(regions, key=_area, reverse=True):
        inside = [box for box in boxes if _covered_by(box, region) >= 0.8]
        if len(inside) < 2:
            continue

        # A region spanning two prints contains two photo-sized pieces; a
        # single print, however faded, holds at most one.
        photo_sized = [
            box
            for box in inside
            if _area(box) >= _FRAGMENT_DOMINANCE * _area(region)
        ]
        if len(photo_sized) > 1:
            continue

        boxes = [box for box in boxes if box not in inside]
        boxes.append(region)
    return boxes


def _paper_regions(
    bgr: np.ndarray, mask: np.ndarray | None = None
) -> list[tuple[int, int, int, int]]:
    """Tight paper regions: whole prints, white margins included."""
    if mask is None:
        mask = _photo_mask(bgr, 240)
    cleaned = _clean_mask(mask)

    height, width = cleaned.shape[:2]
    total = float(height * width)
    if total <= 0:
        return []

    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    regions = []
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < 0.01 * total:
            continue
        component = np.where(labels == index, 255, 0).astype(np.uint8)
        component = _trim_appendages(component)
        ys, xs = np.nonzero(component)
        if ys.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        if (x1 - x0) * (y1 - y0) < 0.01 * total:
            continue
        regions.append((x0, y0, x1 - x0, y1 - y0))

    # A noisy background (or one that is not clipped white) makes everything
    # look like paper. Trust nothing in that case.
    covered = sum(_area(region) for region in regions)
    if covered > _PAPER_REGION_COVERAGE * total:
        return []
    return regions


def _area(box: tuple[int, int, int, int]) -> int:
    return box[2] * box[3]


def _intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[0] + a[2], b[0] + b[2])
    y1 = min(a[1] + a[3], b[1] + b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def _covered_by(
    box: tuple[int, int, int, int], other: tuple[int, int, int, int]
) -> float:
    return _intersection(box, other) / max(1, _area(box))


def _covers_sheet(
    rect: tuple[tuple[float, float], tuple[float, float], float],
    width: int,
    height: int,
) -> bool:
    points = cv2.boxPoints(rect)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return (x1 - x0) * (y1 - y0) >= 0.95 * width * height


def _edge_ring(mask: np.ndarray) -> int:
    """Estimate the thickness of a dark frame hugging the sheet border."""
    def run(rows: np.ndarray) -> int:
        thickness = 0
        for row in rows:
            if np.count_nonzero(row) / row.size < 0.9:
                break
            thickness += 1
        return thickness

    top = run(mask)
    bottom = run(mask[::-1])
    left = run(mask.T)
    right = run(mask.T[::-1])
    return max(top, bottom, left, right)


def _ink_density(
    mask: np.ndarray,
    rect: tuple[tuple[float, float], tuple[float, float], float],
) -> float:
    """Fraction of foreground pixels inside the rectangle's bounding box."""
    x, y, w, h = cv2.boundingRect(cv2.boxPoints(rect).astype(np.int32))
    x = max(0, x)
    y = max(0, y)
    w = min(w, mask.shape[1] - x)
    h = min(h, mask.shape[0] - y)
    if w <= 0 or h <= 0:
        return 0.0
    patch = mask[y : y + h, x : x + w]
    return float(cv2.countNonZero(patch)) / float(w * h)


def _deskew_geometry(
    rect: tuple[tuple[float, float], tuple[float, float], float]
) -> tuple[tuple[float, float], tuple[float, float], float, tuple[tuple[float, float], ...]]:
    """Turn a min-area rect into center, (width, height), small angle and corners.

    ``cv2.minAreaRect`` reports angles in a way that swaps width/height for
    portrait photos. Deriving the angle from the rectangle edges keeps the
    original orientation: the correction is always within +/-45 degrees and
    the edge closest to horizontal defines the photo width.
    """
    points = cv2.boxPoints(rect).astype(np.float64)
    edge_a = points[1] - points[0]
    edge_b = points[2] - points[1]

    angle_a = _horizontal_angle(edge_a)
    angle_b = _horizontal_angle(edge_b)

    if abs(angle_a) <= abs(angle_b):
        angle = angle_a
        width = float(np.hypot(*edge_a))
        height = float(np.hypot(*edge_b))
    else:
        angle = angle_b
        width = float(np.hypot(*edge_b))
        height = float(np.hypot(*edge_a))

    center = (float(points[:, 0].mean()), float(points[:, 1].mean()))
    corners = tuple((float(x), float(y)) for x, y in points)
    return center, (width, height), angle, corners


def _horizontal_angle(edge: np.ndarray) -> float:
    """Angle of an edge in degrees, normalised to (-90, 90]."""
    angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
    if angle > 90.0:
        angle -= 180.0
    elif angle <= -90.0:
        angle += 180.0
    return angle


def _extract_photo(
    bgr: np.ndarray,
    *,
    center: tuple[float, float],
    size: tuple[float, float],
    angle: float,
    corners: tuple[tuple[float, float], ...],
    padding: int,
    deskew: bool,
    trim: bool,
    trim_threshold: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop the rotated rectangle from the sheet, keeping orientation."""
    width = max(1, int(round(size[0])))
    height = max(1, int(round(size[1])))
    left = int(round(center[0] - size[0] / 2.0))
    top = int(round(center[1] - size[1] / 2.0))

    if deskew and abs(angle) > 0.05:
        crop = _warp_photo(bgr, center, size, angle)
    elif abs(angle) > 0.05:
        # Keep the scanned tilt: crop the whole bounding box so no corner is cut.
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        left = int(np.floor(min(xs)))
        top = int(np.floor(min(ys)))
        width = max(1, int(np.ceil(max(xs))) - left)
        height = max(1, int(np.ceil(max(ys))) - top)
        crop = _clip(bgr, left, top, width, height)
        # Trimming would cut the photo's tilted corners here.
        trim = False
    else:
        crop = _clip(bgr, left, top, width, height)

    box = _clamped_box(left, top, width, height, bgr.shape)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if crop.size:
        offset_y = (height - crop.shape[0]) // 2
        offset_x = (width - crop.shape[1]) // 2
        canvas[
            offset_y : offset_y + crop.shape[0],
            offset_x : offset_x + crop.shape[1],
        ] = crop

    if trim:
        canvas = _trim_margins(canvas, trim_threshold)

    if padding:
        canvas = cv2.copyMakeBorder(
            canvas,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=_WHITE_BGR,
        )
    return canvas, box


def _trim_margins(image: np.ndarray, bg_level: int) -> np.ndarray:
    """Cut uniform bright bands from the border (shadow, antialiased edge)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if height < 4 or width < 4:
        return image

    limit_y = min(height // 2, max(8, int(round(height * 0.03))))
    limit_x = min(width // 2, max(8, int(round(width * 0.03))))

    top = _background_run(gray, bg_level, limit_y)
    bottom = _background_run(gray[::-1], bg_level, limit_y)
    left = _background_run(gray.T, bg_level, limit_x)
    right = _background_run(gray.T[::-1], bg_level, limit_x)

    if not (top or bottom or left or right):
        return image
    top = min(top, height - 1)
    bottom = min(bottom, height - 1 - top)
    left = min(left, width - 1)
    right = min(right, width - 1 - left)
    return image[top : height - bottom, left : width - right]


def _background_run(lines: np.ndarray, bg_level: int, limit: int) -> int:
    count = 0
    for line in lines:
        if count >= limit or not _is_background_line(line, bg_level):
            break
        count += 1
    return count


def _is_background_line(line: np.ndarray, bg_level: int) -> bool:
    """True for a border line that is mostly background, not photo content."""
    bright = float(np.count_nonzero(line >= bg_level)) / line.size
    return bright >= 0.35


def _warp_photo(
    bgr: np.ndarray,
    center: tuple[float, float],
    size: tuple[float, float],
    angle: float,
) -> np.ndarray:
    """Rotate a single photo to axis alignment, warping only its neighbourhood."""
    cx, cy = center
    half_w, half_h = size[0] / 2.0, size[1] / 2.0
    width = max(1, int(round(size[0])))
    height = max(1, int(round(size[1])))

    radians = np.deg2rad(angle)
    cos_a, sin_a = float(np.cos(radians)), float(np.sin(radians))

    # Source-space corners of the photo (inverse rotation of the output rect).
    dx = np.array([-half_w, half_w, half_w, -half_w])
    dy = np.array([-half_h, -half_h, half_h, half_h])
    source_x = cx + cos_a * dx - sin_a * dy
    source_y = cy + sin_a * dx + cos_a * dy

    x0 = max(0, int(np.floor(source_x.min())) - 2)
    y0 = max(0, int(np.floor(source_y.min())) - 2)
    x1 = min(bgr.shape[1], int(np.ceil(source_x.max())) + 2)
    y1 = min(bgr.shape[0], int(np.ceil(source_y.max())) + 2)

    # Map source-global coordinates to an output canvas that starts exactly at
    # the photo's top-left corner. Account for both the source ROI offset and
    # the destination canvas offset.
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    matrix[0, 2] += cos_a * x0 + sin_a * y0 - (cx - half_w)
    matrix[1, 2] += -sin_a * x0 + cos_a * y0 - (cy - half_h)

    region = bgr[y0:y1, x0:x1]
    return cv2.warpAffine(
        region,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_WHITE_BGR,
    )


def _clip(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """Crop with bounds clamping."""
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(image.shape[1], x + width)
    y1 = min(image.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return image[0:0, 0:0]
    return image[y0:y1, x0:x1]


def _clamped_box(
    x: int, y: int, width: int, height: int, shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """Clamp a box to the image and return it in x, y, w, h form."""
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(shape[1], x + width)
    y1 = min(shape[0], y + height)
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def _reading_order(photos: list[DetectedPhoto]) -> list[DetectedPhoto]:
    """Top-to-bottom, then left-to-right, grouping photos that share a row."""
    if len(photos) <= 1:
        return photos

    ordered = sorted(photos, key=lambda p: (p.box[1] + p.box[3] / 2.0, p.box[0]))
    heights = sorted(p.box[3] for p in ordered)
    median_height = max(1.0, float(heights[len(heights) // 2]))
    tolerance = median_height * 0.5

    rows: list[list[DetectedPhoto]] = []
    row_reference = None
    for photo in ordered:
        center_y = photo.box[1] + photo.box[3] / 2.0
        if row_reference is None or abs(center_y - row_reference) > tolerance:
            rows.append([photo])
            row_reference = center_y
        else:
            rows[-1].append(photo)

    result: list[DetectedPhoto] = []
    for row in rows:
        result.extend(sorted(row, key=lambda p: p.box[0]))
    return result
