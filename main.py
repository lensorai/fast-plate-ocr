#!/usr/bin/env python3
"""
Run license plate OCR inference on all images in a folder and print results.
"""

import argparse
import sys
from pathlib import Path

from fast_plate_ocr import LicensePlateRecognizer


# Common image extensions supported by the library (OpenCV)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def collect_image_paths(folder: Path) -> list[Path]:
    """Collect all image file paths from a folder, sorted by name."""
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    paths = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: p.name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run license plate OCR on all images in a folder and print results."
    )
    default_folder = Path(
        "/Users/anwesh.marwade@pon.com/repos/ai-library/src/ai_library/assets/dekra_images_debug/dekra_images_crop"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=default_folder,
        help="Path to folder containing plate images",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="cct-s-v2-global-model",
        help="Model name from the HUB (default: cct-s-v2-global-model)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        action="store_true",
        help="Include per-character confidence in output",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto)",
    )
    args = parser.parse_args()

    folder: Path = args.folder.resolve()
    try:
        image_paths = collect_image_paths(folder)
    except NotADirectoryError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not image_paths:
        print(f"No image files found in {folder}", file=sys.stderr)
        print(f"Supported extensions: {', '.join(sorted(IMAGE_EXTENSIONS))}", file=sys.stderr)
        return 1

    print(f"Loading model '{args.model}'...")
    recognizer = LicensePlateRecognizer(
        hub_ocr_model=args.model,
        device=args.device,
    )
    print(f"Running inference on {len(image_paths)} image(s)...\n")

    # Run on all images (list of paths is supported)
    path_strs = [str(p) for p in image_paths]
    predictions = recognizer.run(
        path_strs,
        return_confidence=args.confidence,
    )

    # Print results
    for path, pred in zip(image_paths, predictions, strict=True):
        line = f"{path.name}\t{pred.plate}"
        if pred.region is not None:
            line += f"\tregion={pred.region}"
            if pred.region_prob is not None:
                line += f" ({pred.region_prob:.2f})"
        if args.confidence and pred.char_probs is not None and pred.char_probs.size:
            avg_conf = float(pred.char_probs.mean())
            line += f"\tconf={avg_conf:.2f}"
        print(line)

    return 0


if __name__ == "__main__":
    main()
