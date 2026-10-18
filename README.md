# Vibecoded Photo Scan Normalizer

Flatbed scans often contain several photos on one sheet. `photosplit` finds
each photo on the white background, straightens small rotations, turns
sideways and upside-down prints the right way up, crops it tightly and writes
it to its own file.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # creates .venv with only this project's dependencies
```

`uv.lock` pins the exact dependency versions.

## Usage

```bash
# One scanned sheet -> out/scan_01.png, out/scan_02.png, ...
uv run photosplit scan.png

# Choose the output format (files still go to out/)
uv run photosplit scan.png --format jpg

# Process a whole folder (optionally recursive), 8 sheets at a time
uv run photosplit scans/ --recursive -j 8

# Keep the scanned orientation and just split
uv run photosplit scan.png --rotate none

# Add a white border, keep the scanned angles, write a debug overlay
uv run photosplit scan.png --padding 24 --no-deskew --debug

# Keep a photo's own white border (disable automatic trimming)
uv run photosplit scan.png --no-trim
```

Every photo is named `<input name>_NN.<format>` and numbered in reading order
(top-to-bottom, then left-to-right). Output files are PNG by default; use
`--format jpg` or `--format webp` for smaller files.

## Orientation

Photos are turned upright automatically. A print can lie on the glass in any
of four orientations and nothing in its outline says which edge is the top, so
the decision is made from the picture itself:

- **Faces** are the main evidence. A small bundled face detector (YuNet, MIT,
  ~230 KB) scores each of the four rotations, and the one where faces look
  most convincing wins. This covers the overwhelming majority of snapshots.
- **Colour** is a fallback for photos without faces: a bright blue sky sits
  above darker ground.
- **Sheet consensus**: photos on one sheet were scanned in one pass, so when
  most of them are confidently sideways, the others follow.

When the evidence is inconclusive the photo is left as scanned rather than
guessed at — a sideways photo is obvious, an upside-down one is easy to miss.
Every run prints the decision per photo (`[90°, 90°, ?]`, where `?` means
undecided), and `--rotate 0|90|180|270` forces a rotation, `--rotate none`
disables the whole step.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `-o, --output DIR` | `out` | Output directory. |
| `--format {png,jpg,webp}` | `png` | Output file format. |
| `--threshold 0-255` | `240` | Pixels at/above this brightness are background. Lower it for grey paper, raise it for off-white scans. |
| `--min-area RATIO` | `0.002` | Ignore blobs smaller than this fraction of the sheet (removes dust). |
| `--padding PX` | `0` | White border around each photo. |
| `--no-deskew` | off | Keep the scanned rotation. |
| `--no-trim` | off | Keep leftover bright borders around each photo. |
| `--trim-threshold 0-255` | `--threshold` − 10 | Brightness at which a border line counts as background and is trimmed. |
| `--rotate {auto,none,0,90,180,270}` | `auto` | Detect the upright orientation, keep the scanned one, or force a clockwise rotation. |
| `-j, --jobs N` | auto | Sheets to process in parallel (`0` = automatic, `1` = serial). |
| `--skip-existing` | off | Leave output files that already exist alone, so interrupted batches can be resumed. |
| `--recursive` | off | Search input directories recursively. |
| `--debug` | off | Write `<name>_detections.png` with the detected outlines. |
| `--dry-run` | off | Detect and report only; write nothing. |
| `-q, --quiet` | off | Only warnings and errors. |


## Notes

- Photos must not overlap or touch each other; keep a visible white gap between
  them on the scanner glass. A narrow shadowed gap between two prints is still
  detected as a gap, so prints that almost touch are split apart correctly.
- Faded, overexposed and black-and-white prints are handled: the splitter also
  looks for the print's *paper*, which is never quite the same white as the
  scanner bed. A photo that is mostly white areas is kept as one photo instead
  of being cut into pieces at every bright band.
- Thin scanner shadows and streaks along the sheet edge are ignored instead of
  being reported as a thin extra photo, and pale fragments of one print are
  rejoined even when an overexposed band separates them.
- Overlapping duplicate outlines (a bright print with a dark inner frame, for
  example) are reported once, and outlines that reach beyond the sheet are
  discarded as scanner-shadow artifacts.
- White areas *inside* a photo are handled. Uniform bright bands left over from
  scanner shadows or soft photo edges are trimmed automatically; if a photo has
  a genuinely white border that you want to keep, pass `--no-trim`.
- Extremely pale photo edges may be trimmed by a few pixels. Raise
  `--threshold` to keep pale edges; lower it if grey paper or a shadow is
  detected as a photo.
- Photos that are already straight and upright are cropped without any
  resampling, so their pixels are untouched. Rotations by 90 degrees are pure
  pixel permutations; only deskewing resamples.

## Tests

```bash
uv run pytest
```
