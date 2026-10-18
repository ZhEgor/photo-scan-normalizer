"""Command-line entry point for photosplit."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .core import DetectedPhoto, discover_images, load_image, save_image, split_file

_FORMATS = {"png": ".png", "jpg": ".jpg", "webp": ".webp"}

# Process pools are only worth their memory when there is real work to spread.
_DEFAULT_JOBS = min(8, os.cpu_count() or 1)


@dataclass(frozen=True)
class _Options:
    output: Path
    suffix: str
    threshold: int
    min_area: float
    padding: int
    deskew: bool
    trim: bool
    trim_threshold: int | None
    rotate: str
    debug: bool
    dry_run: bool
    skip_existing: bool


@dataclass
class _SheetResult:
    source: Path
    written: list[Path] = field(default_factory=list)
    skipped: int = 0
    rotations: list[tuple[int, float, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    error: str | None = None
    warning: str | None = None
    debug_error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photosplit",
        description=(
            "Split scanned sheets that contain several photos on a white "
            "background into separate, straight, upright image files."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="INPUT",
        help="image file(s) or directory(ies) to process",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="out",
        metavar="DIR",
        help="output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--format",
        choices=sorted(_FORMATS),
        default="png",
        help="output file format (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=240,
        metavar="0-255",
        help=(
            "pixel brightness at or above which is background; lower for "
            "greyish paper, raise for off-white scans (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.002,
        metavar="RATIO",
        help=(
            "ignore blobs smaller than this fraction of the sheet, which "
            "filters dust and specks (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        metavar="PX",
        help="white border around each extracted photo (default: %(default)s)",
    )
    parser.add_argument(
        "--no-deskew",
        action="store_true",
        help="keep the photos at their scanned angle",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="keep leftover bright borders around each photo",
    )
    parser.add_argument(
        "--trim-threshold",
        type=int,
        default=None,
        metavar="0-255",
        help=(
            "brightness at which a border line is considered background; "
            "defaults to --threshold minus 10"
        ),
    )
    parser.add_argument(
        "--rotate",
        choices=("auto", "none", "0", "90", "180", "270"),
        default="auto",
        help=(
            "upright orientation: auto detects sideways and upside-down prints "
            "from their content, none keeps the scanned orientation, a number "
            "forces one clockwise rotation (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=0,
        metavar="N",
        help=(
            "sheets to process in parallel; 0 picks a sensible number for this "
            "machine, 1 disables parallelism (default: 0)"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="leave output files that already exist alone (resumable runs)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="search input directories recursively",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="also write a copy of each sheet with detected photo outlines",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="detect photos but do not write any files",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="only print warnings and errors",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _quiet_opencv()
    if not 0 <= args.threshold <= 255:
        print("error: --threshold must be between 0 and 255", file=sys.stderr)
        return 2
    if args.padding < 0:
        print("error: --padding must not be negative", file=sys.stderr)
        return 2
    if not 0 < args.min_area < 1:
        print("error: --min-area must be between 0 and 1", file=sys.stderr)
        return 2
    if args.trim_threshold is not None and not 0 <= args.trim_threshold <= 255:
        print("error: --trim-threshold must be between 0 and 255", file=sys.stderr)
        return 2
    if args.jobs < 0:
        print("error: --jobs must not be negative", file=sys.stderr)
        return 2

    try:
        inputs = discover_images(args.inputs, recursive=args.recursive)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output)
    inputs = [p for p in inputs if not _is_inside(p, output_dir)]
    if not inputs:
        print("error: no input images found", file=sys.stderr)
        return 1

    options = _Options(
        output=output_dir,
        suffix=_FORMATS[args.format],
        threshold=args.threshold,
        min_area=args.min_area,
        padding=args.padding,
        deskew=not args.no_deskew,
        trim=not args.no_trim,
        trim_threshold=args.trim_threshold,
        rotate=args.rotate,
        debug=args.debug,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )

    tasks = [(source, options) for source in inputs]
    jobs = _job_count(args.jobs, len(tasks))

    total = 0
    skipped = 0
    failures = 0
    rotations = {0: 0, 90: 0, 180: 0, 270: 0}
    undecided = 0

    for result in _run(tasks, jobs):
        if result.error is not None:
            print(f"error: {result.source}: {result.error}", file=sys.stderr)
            failures += 1
            continue

        if result.warning is not None:
            print(f"warning: {result.source}: {result.warning}", file=sys.stderr)
            continue

        for message in result.errors:
            print(f"error: {message}", file=sys.stderr)
            failures += 1

        if result.debug_error is not None:
            print(
                f"warning: {result.source}: debug overlay failed: "
                f"{result.debug_error}",
                file=sys.stderr,
            )

        for rotation, _confidence, basis in result.rotations:
            if basis == "none":
                undecided += 1
            else:
                rotations[rotation % 360] += 1

        count = len(result.written) + result.skipped
        total += count
        skipped += result.skipped

        if not args.quiet:
            verb = "would write" if args.dry_run else "wrote"
            names = ", ".join(str(path) for path in result.written)
            line = f"{result.source}: {count} photo(s), {verb} {names}"
            if result.skipped:
                line += f" ({result.skipped} already present)"
            if args.rotate != "none":
                line += f" {_format_rotations(result.rotations)}"
            print(line)

    if not args.quiet:
        print(f"done: {total} photo(s) from {len(inputs)} sheet(s) -> {output_dir}")
        if args.rotate != "none":
            rotated = sum(rotations.values()) - rotations[0]
            parts = [
                f"{angle}°: {rotations[angle]}"
                for angle in (90, 180, 270)
                if rotations[angle]
            ]
            summary = f"orientation: {rotated} rotated"
            if parts:
                summary += f" ({', '.join(parts)})"
            summary += f", {rotations[0]} already upright, {undecided} undecided"
            if skipped:
                summary += f", {skipped} skipped"
            print(summary)
    return 1 if failures else 0


def _job_count(requested: int, tasks: int) -> int:
    if tasks <= 1:
        return 1
    if requested > 0:
        return min(requested, tasks)
    return min(_DEFAULT_JOBS, tasks)


def _run(tasks: list[tuple[Path, _Options]], jobs: int) -> list[_SheetResult]:
    if jobs <= 1:
        return [_process_sheet(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as pool:
        # map() keeps results in input order, so output stays deterministic.
        return list(pool.map(_process_sheet, tasks, chunksize=1))


def _init_worker() -> None:
    """Let processes do the parallel work instead of OpenCV's own thread pool."""
    cv2.setNumThreads(1)
    _quiet_opencv()


def _quiet_opencv() -> None:
    """Hide OpenCV's harmless dnn backend notice printed on start-up."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:  # pragma: no cover - older OpenCV builds
        pass


def _process_sheet(task: tuple[Path, _Options]) -> _SheetResult:
    """Split one sheet and write its photos. Safe to run in another process."""
    source, options = task
    result = _SheetResult(source)

    try:
        photos = split_file(
            source,
            white_threshold=options.threshold,
            min_area_ratio=options.min_area,
            padding=options.padding,
            deskew=options.deskew,
            trim=options.trim,
            trim_threshold=options.trim_threshold,
            rotate=options.rotate,
        )
    except Exception as exc:  # one broken scan must not abort a whole batch
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    if not photos:
        result.warning = (
            "no photos detected (try adjusting --threshold or --min-area)"
        )
        return result

    width = max(2, len(str(len(photos))))
    for index, photo in enumerate(photos, start=1):
        destination = (
            options.output / f"{source.stem}_{index:0{width}d}{options.suffix}"
        )
        result.rotations.append(
            (photo.rotation, photo.rotation_confidence, photo.rotation_basis)
        )

        if options.skip_existing and destination.exists():
            result.skipped += 1
            continue
        if options.dry_run:
            result.written.append(destination)
            continue
        try:
            save_image(photo.image, destination)
            result.written.append(destination)
        except (OSError, ValueError) as exc:
            result.errors.append(f"{destination}: {exc}")

    if options.debug and not options.dry_run:
        try:
            _write_debug(source, photos, options.output)
        except (OSError, ValueError) as exc:
            result.debug_error = str(exc)

    return result


def _format_rotations(rotations: list[tuple[int, float, str]]) -> str:
    labels = [
        "?" if basis == "none" else f"{rotation % 360}°"
        for rotation, _confidence, basis in rotations
    ]
    return "[" + ", ".join(labels) + "]"


def _is_inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _write_debug(
    source: Path, photos: list[DetectedPhoto], output_dir: Path
) -> None:
    """Save the sheet with detected photo outlines for troubleshooting."""
    canvas = load_image(source)
    for index, photo in enumerate(photos, start=1):
        points = np.array(photo.corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, (0, 200, 0), 3, cv2.LINE_AA)
        anchor = tuple(np.array(photo.corners, dtype=int)[0])
        label = str(index)
        if photo.rotation:
            label += f" {photo.rotation}deg"
        cv2.putText(
            canvas,
            label,
            (int(anchor[0]) + 6, int(anchor[1]) + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    destination = output_dir / f"{source.stem}_detections.png"
    save_image(canvas, destination)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
