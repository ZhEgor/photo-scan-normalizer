"""Content-based upright orientation for scanned photos.

Deskewing only recovers the small angle a print was laid down with. Whether a
print is sideways or upside down is a question about its *content*: geometry
has no idea which edge is the top. This module settles that with a small vote
over the four 90-degree rotations.

Faces carry almost all of the signal, so they are the primary evidence: a
network detects faces noticeably better in the correct orientation than in the
wrong one. The rotation where faces look most convincing wins. Photos with no
faces fall back to coarse colour statistics (bright blue sky above darker
ground) and, failing that, to the orientation of the other photos on the same
sheet. Everything stays conservative: when the evidence is thin the photo is
left as scanned rather than guessed at, because a wrongly rotated photo is
harder to notice than an obviously sideways one.
"""

from __future__ import annotations

import threading
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

__all__ = [
    "ROTATIONS",
    "Orientation",
    "apply_rotation",
    "estimate_orientation",
    "estimate_orientations",
]

ROTATIONS: tuple[int, ...] = (0, 90, 180, 270)

_MODEL_NAME = "face_detection_yunet_2023mar.onnx"

# Long side used when scanning a photo (or, as a rescue, the whole sheet) for
# faces. Faces are large on a 300 dpi scan, so a modest size is plenty.
_DETECT_SIDE_PHOTO = 640
_DETECT_SIDE_SHEET = 1400

# YuNet scores: correct orientations land around 0.85-0.95, rotated ones are
# either absent or degraded (sideways faces are usually not detected at all,
# upside-down ones only partially). 0.5 is the model's own threshold.
_FACE_FLOOR = 0.5
_FACE_STRONG = 0.85
_MIN_FACE_WEIGHT = 0.25

# A rotation needs a face that is actually convincing, a clear lead over the
# runner-up, and must not be beaten by a competitor at 180 degrees, where a
# face can still be detected while upside down.
_MIN_SCORE = 0.015
_MIN_TOP = 0.55
_MIN_MARGIN = 1.2
# Rescuing a photo from the whole-sheet pass needs stronger evidence: the
# sheet is downscaled, so a weak detection there is usually noise.
_RESCUE_MIN_TOP = 0.85
_SHEET_CONFIDENCE = 0.5

# Colour fallback: sky is blue and bright, ground is not.
_COLOUR_SIDE = 320
_COLOUR_MIN_SCORE = 10.0
_COLOUR_MIN_MARGIN = 1.6


class _Face(NamedTuple):
    cx: float
    cy: float
    area: float
    score: float


class _Evidence(NamedTuple):
    """What the face detector saw in each of the four rotations."""

    scores: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)  # area-weighted strength
    tops: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)  # best face confidence
    counts: tuple[int, ...] = (0, 0, 0, 0)
    strongs: tuple[int, ...] = (0, 0, 0, 0)


@dataclass(frozen=True)
class Orientation:
    """The decided rotation of one photo and why it was chosen."""

    rotation: int = 0  # clockwise degrees: 0, 90, 180 or 270
    confidence: float = 0.0
    basis: str = "none"  # faces | sheet | colour | none
    scores: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def decided(self) -> bool:
        return self.basis != "none"


def apply_rotation(image: np.ndarray, rotation: int) -> np.ndarray:
    """Rotate by a multiple of 90 degrees without resampling a single pixel."""
    rotation = int(rotation) % 360
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation != 0:
        raise ValueError(f"rotation must be 0, 90, 180 or 270: {rotation}")
    return image


def estimate_orientation(image: np.ndarray) -> Orientation:
    """Decide the upright rotation of a single photo."""
    evidence = _image_evidence(image)
    rotation = _decide(evidence)
    if rotation is not None:
        return _result(rotation, evidence, "faces")

    if not any(evidence.counts):
        colour = _colour_evidence(image)
        rotation = _decide_colour(colour)
        if rotation is not None:
            return _result_from_scores(rotation, colour, "colour")

    return Orientation(scores=evidence.scores)


def estimate_orientations(sheet: np.ndarray, photos: list) -> list[Orientation]:
    """Decide one orientation per photo, in the order the photos are given.

    Each photo is judged on its own crop first: there the faces are as large
    as they will ever get. The whole sheet is only scanned again for photos
    that stay undecided, which rescues faces that were cropped at a photo's
    edge. ``photos`` only needs ``.box`` and ``.image`` attributes.
    """
    if not photos:
        return []

    decisions: list[Orientation | None] = []
    unresolved: list[int] = []
    for index, photo in enumerate(photos):
        evidence = _image_evidence(photo.image)
        rotation = _decide(evidence)
        if rotation is None:
            decisions.append(None)
            unresolved.append(index)
        else:
            decisions.append(_result(rotation, evidence, "faces"))

    if unresolved:
        faces = _sheet_faces(sheet)
        for index in unresolved:
            photo = photos[index]
            evidence = _box_evidence(faces, photo.box)
            rotation = _decide(evidence, min_top=_RESCUE_MIN_TOP)
            if rotation is not None:
                decisions[index] = _result(rotation, evidence, "faces")
                continue

            colour = _colour_evidence(photo.image)
            rotation = _decide_colour(colour)
            if rotation is not None:
                decisions[index] = _result_from_scores(rotation, colour, "colour")
            else:
                decisions[index] = Orientation(scores=evidence.scores)

    settled = [
        decision if decision is not None else Orientation() for decision in decisions
    ]
    return _apply_sheet_consensus(settled)


def _result(rotation: int, evidence: _Evidence, basis: str) -> Orientation:
    best = ROTATIONS.index(rotation % 360)
    runner_up = max(evidence.tops[i] for i in range(len(ROTATIONS)) if i != best)
    top = evidence.tops[best]
    confidence = 1.0 if runner_up <= 0 else 1.0 - runner_up / top
    return Orientation(
        rotation=rotation,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        basis=basis,
        scores=evidence.scores,
    )


def _result_from_scores(rotation: int, scores: list[float], basis: str) -> Orientation:
    best = ROTATIONS.index(rotation % 360)
    runner_up = max(scores[i] for i in range(len(ROTATIONS)) if i != best)
    confidence = 1.0 if runner_up <= 0 else 1.0 - runner_up / max(scores[best], 1e-6)
    return Orientation(
        rotation=rotation,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        basis=basis,
        scores=tuple(float(s) for s in scores),
    )


def _apply_sheet_consensus(decisions: list[Orientation]) -> list[Orientation]:
    """Adopt the sheet's majority orientation for photos that abstained.

    A sheet is scanned in one pass, so every print on it was laid down in a
    similar way. This only fills in photos where the photo's own evidence was
    inconclusive; confident per-photo decisions always win.
    """
    decided = [d.rotation for d in decisions if d.decided]
    if len(decisions) < 2 or len(decided) < 2:
        return decisions

    rotation, hits = Counter(decided).most_common(1)[0]
    if hits < max(2, -(-2 * len(decided) // 3)):  # two thirds
        return decisions

    return [
        d
        if d.decided
        else Orientation(
            rotation=rotation,
            confidence=_SHEET_CONFIDENCE,
            basis="sheet",
            scores=d.scores,
        )
        for d in decisions
    ]


def _decide(evidence: _Evidence, *, min_top: float = _MIN_TOP) -> int | None:
    """Pick a rotation from face evidence, or ``None`` when it is too thin."""
    if max(evidence.scores) < _MIN_SCORE:
        return None

    # A rotation with a genuinely strong face beats one without; after that
    # the single most convincing face wins, and finally the overall weight.
    best = max(
        range(len(ROTATIONS)),
        key=lambda i: (
            evidence.strongs[i] > 0,
            evidence.tops[i],
            evidence.scores[i],
        ),
    )
    if evidence.tops[best] < min_top:
        return None

    others = [i for i in range(len(ROTATIONS)) if i != best]
    second = max(others, key=lambda i: (evidence.strongs[i] > 0, evidence.tops[i]))

    if evidence.strongs[best] > 0 and evidence.strongs[second] == 0:
        return ROTATIONS[best]

    margin = evidence.tops[best] / max(evidence.tops[second], 1e-6)
    if margin < _MIN_MARGIN:
        return None
    return ROTATIONS[best]


def _decide_colour(scores: list[float]) -> int | None:
    if not scores or max(scores) < _COLOUR_MIN_SCORE:
        return None
    best = int(np.argmax(scores))
    second = max(scores[i] for i in range(len(scores)) if i != best)
    if scores[best] / max(second, 1.0) < _COLOUR_MIN_MARGIN:
        return None
    return ROTATIONS[best]


def _image_evidence(image: np.ndarray) -> _Evidence:
    """Detect faces in each of the four rotations of a downscaled photo."""
    if min(image.shape[:2]) < 16:
        return _Evidence()
    small = _resize(image, _DETECT_SIDE_PHOTO)
    area = float(max(1, small.shape[0] * small.shape[1]))
    scores = []
    tops = []
    counts = []
    strongs = []
    for rotation in ROTATIONS:
        score, top, count, strong = _aggregate(
            _detect(apply_rotation(small, rotation)), area
        )
        scores.append(score)
        tops.append(top)
        counts.append(count)
        strongs.append(strong)
    return _Evidence(tuple(scores), tuple(tops), tuple(counts), tuple(strongs))


def _sheet_faces(
    sheet: np.ndarray, max_side: int = _DETECT_SIDE_SHEET
) -> dict[int, list[_Face]]:
    """Faces found in each rotation of the sheet, mapped back to sheet space."""
    height, width = sheet.shape[:2]
    if min(height, width) < 16:
        return {rotation: [] for rotation in ROTATIONS}

    small = _resize(sheet, max_side)
    scale_x = width / small.shape[1]
    scale_y = height / small.shape[0]

    found: dict[int, list[_Face]] = {}
    for rotation in ROTATIONS:
        rotated = apply_rotation(small, rotation)
        faces = []
        for face in _detect(rotated):
            x, y = _to_source(
                face.cx, face.cy, rotation, small.shape[1], small.shape[0]
            )
            faces.append(
                _Face(
                    cx=x * scale_x,
                    cy=y * scale_y,
                    area=face.area * scale_x * scale_y,
                    score=face.score,
                )
            )
        found[rotation] = faces
    return found


def _to_source(
    x: float, y: float, rotation: int, src_width: int, src_height: int
) -> tuple[float, float]:
    """Map a point from a ``cv2.rotate`` result back to the unrotated image."""
    if rotation == 90:  # clockwise: top-left moves to the top-right
        return float(y), float(src_height - 1 - x)
    if rotation == 180:
        return float(src_width - 1 - x), float(src_height - 1 - y)
    if rotation == 270:  # counter-clockwise: top-left moves to the bottom-left
        return float(src_width - 1 - y), float(x)
    return float(x), float(y)


def _box_evidence(
    faces_by_rotation: dict[int, list[_Face]],
    box: tuple[int, int, int, int],
) -> _Evidence:
    x, y, width, height = box
    area = float(max(1, width * height))
    scores = []
    tops = []
    counts = []
    strongs = []
    for rotation in ROTATIONS:
        inside = [
            face
            for face in faces_by_rotation[rotation]
            if x - 2 <= face.cx <= x + width + 2
            and y - 2 <= face.cy <= y + height + 2
        ]
        score, top, count, strong = _aggregate(inside, area)
        scores.append(score)
        tops.append(top)
        counts.append(count)
        strongs.append(strong)
    return _Evidence(tuple(scores), tuple(tops), tuple(counts), tuple(strongs))


def _aggregate(faces: list[_Face], area: float) -> tuple[float, float, int, int]:
    score = 0.0
    top = 0.0
    strong = 0
    for face in faces:
        top = max(top, face.score)
        if face.score >= _FACE_STRONG:
            strong += 1
        # Big faces are more reliable than background clutter.
        weight = min(1.0, max(_MIN_FACE_WEIGHT, (face.area / max(area, 1.0)) ** 0.5))
        score += max(0.0, face.score - _FACE_FLOOR) * weight
    return score, top, len(faces), strong


def _colour_evidence(image: np.ndarray) -> list[float]:
    """Score each rotation by how much sky sits above how much ground."""
    if min(image.shape[:2]) < 8:
        return [0.0, 0.0, 0.0, 0.0]
    small = _resize(image, _COLOUR_SIDE)
    scores = []
    for rotation in ROTATIONS:
        rotated = apply_rotation(small, rotation)
        half = rotated.shape[0] // 2
        top = rotated[:half].astype(np.float32)
        bottom = rotated[half:].astype(np.float32)
        if top.size == 0 or bottom.size == 0:
            scores.append(0.0)
            continue
        blue = float(top[:, :, 0].mean() - bottom[:, :, 0].mean())
        bright = float(top.mean() - bottom.mean())
        scores.append(0.7 * blue + 0.3 * bright)
    return scores


def _resize(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    factor = max_side / max(height, width)
    if factor >= 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, int(width * factor)), max(1, int(height * factor))),
        interpolation=cv2.INTER_AREA,
    )


_local = threading.local()
_model_warned = False


def _detector() -> cv2.FaceDetectorYN | None:
    """One YuNet instance per thread; the detector keeps mutable input state."""
    global _model_warned
    detector = getattr(_local, "detector", None)
    if detector is not None:
        return detector

    path = _model_path()
    if not path.is_file():
        if not _model_warned:
            _model_warned = True
            warnings.warn(
                f"face model missing ({path}); photo orientation detection is "
                "disabled",
                RuntimeWarning,
                stacklevel=3,
            )
        return None

    detector = cv2.FaceDetectorYN.create(
        str(path), "", (320, 320), _FACE_FLOOR, 0.3, 5000
    )
    _local.detector = detector
    return detector


def _model_path() -> Path:
    return Path(__file__).resolve().parent / "models" / _MODEL_NAME


def _detect(image: np.ndarray) -> list[_Face]:
    if min(image.shape[:2]) < 16:
        return []
    detector = _detector()
    if detector is None:
        return []

    detector.setInputSize((image.shape[1], image.shape[0]))
    try:
        _, faces = detector.detect(image)
    except cv2.error:  # pragma: no cover - defensive, tiny/degenerate frames
        return []
    if faces is None:
        return []
    return [
        _Face(
            cx=float(face[0] + face[2] / 2.0),
            cy=float(face[1] + face[3] / 2.0),
            area=float(face[2] * face[3]),
            score=float(face[-1]),
        )
        for face in faces
    ]
