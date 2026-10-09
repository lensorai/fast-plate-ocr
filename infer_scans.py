#!/usr/bin/env python3
"""
Detect license plates in inspection scans, crop them, and read the plate text.

Each immediate subfolder of the scans root is one inspection. Images in that
folder are frames from the same inspection. A root that contains images and no
scan subfolders is treated as a single inspection.

Detection uses an Ultralytics checkpoint (bounding boxes). Crops are read with
fast-plate-ocr. Results are written as:

- detections.csv: one row per image, or one row per plate box
- scans.csv: one voted plate per inspection
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fast_plate_ocr import LicensePlateRecognizer
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

# Edit these. Matching command-line flags override them.
SCANS_ROOT = Path("/Users/anwesh.marwade@pon.com/Downloads/distrifresh_lp_issues")
DETECTOR = Path(
    "/Users/anwesh.marwade@pon.com/Downloads/lux_models/model-y26s-imgsz_640_vps_lux_license_plate_trailers_v1.pt"
)
OCR_MODEL = "cct-s-v2-global-model"
CONF = 0.80
IOU = 0.30
IMGSZ = 640
CLASSES: list[str] | None = None  # None keeps every detector class
VOTE_CLASS = "license_plate_main"
DEVICE = "mps"  # auto, cpu, mps, or 0
OCR_DEVICE = "auto"  # auto, cpu, or cuda
OCR_BATCH = 32
DETECTOR_BATCH = 1  # ONNX input is fixed at this batch size; short batches are padded
OUTPUT_DIR = Path("distrifresh_lp_issues_out")
SAVE_CROPS = True


@dataclass(frozen=True)
class DetectConfig:
    class_filter: set[str] | None
    imgsz: int
    conf: float
    iou: float
    device: str
    detector_batch: int
    save_crops_dir: Path | None


@dataclass
class DetectionRow:
    scan: str
    image: str
    class_name: str
    det_conf: float | None
    x1: float | None
    y1: float | None
    x2: float | None
    y2: float | None
    plate: str
    char_conf: float | None
    region: str | None
    region_prob: float | None


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name,
    )


def discover_scans(root: Path) -> list[tuple[str, Path]]:
    """Return (scan name, folder) pairs. Subfolders with images win over loose files."""
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    scans = [(d.name, d) for d in sorted(root.iterdir(), key=lambda p: p.name) if d_has_images(d)]
    loose = collect_images(root)
    if scans:
        if loose:
            print(f"Ignoring {len(loose)} image(s) placed directly in {root}. Using {len(scans)} scan folder(s).")
        return scans
    if loose:
        return [(root.name, root)]
    return []


def d_has_images(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith(".") and bool(collect_images(path))


def crop_box(
    image_bgr: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray | None:
    """Crop the plate the way vision-core does: truncate to int, clamp, no border.

    The OCR step then stretches this crop to the model size. Nothing is letterboxed.
    """
    height, width = image_bgr.shape[:2]
    left = max(0, int(x1))
    top = max(0, int(y1))
    right = min(width, int(x2))
    bottom = min(height, int(y2))
    if right <= left or bottom <= top:
        return None
    crop = image_bgr[top:bottom, left:right]
    if crop.size == 0:
        return None
    if crop.ndim == 2:
        return cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def mean_char_conf(char_probs: np.ndarray | None) -> float | None:
    if char_probs is None or char_probs.size == 0:
        return None
    return float(np.mean(char_probs))


def stretch_for_ocr(crop_rgb: np.ndarray, recognizer: LicensePlateRecognizer) -> np.ndarray:
    """Bilinear-stretch a plate crop to the OCR input size, with no padding."""
    config = recognizer.config
    resized = cv2.resize(crop_rgb, (config.img_width, config.img_height), interpolation=cv2.INTER_LINEAR)
    if config.image_color_mode == "grayscale":
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)[..., None]
    return np.ascontiguousarray(resized, dtype=np.uint8)


def read_crops(
    recognizer: LicensePlateRecognizer,
    crops: list[np.ndarray],
    batch_size: int,
) -> list:
    """Run OCR on crops already stretched to the model size.

    A 4-D batch skips the recognizer's own resize, so a letterbox setting in the
    plate config cannot add a border.
    """
    prepared = [stretch_for_ocr(crop, recognizer) for crop in crops]
    predictions = []
    for start in range(0, len(prepared), batch_size):
        batch = np.stack(prepared[start : start + batch_size], axis=0)
        predictions.extend(recognizer.run(batch, return_confidence=True))
    return predictions


def empty_row(scan: str, image: str) -> DetectionRow:
    return DetectionRow(
        scan=scan,
        image=image,
        class_name="",
        det_conf=None,
        x1=None,
        y1=None,
        x2=None,
        y2=None,
        plate="",
        char_conf=None,
        region=None,
        region_prob=None,
    )


def pad_to_batch(paths: list[str], batch_size: int) -> list[str]:
    """Repeat the last path so the length is a multiple of a fixed ONNX batch size."""
    if batch_size <= 1 or not paths:
        return paths
    remainder = len(paths) % batch_size
    if remainder == 0:
        return paths
    return paths + [paths[-1]] * (batch_size - remainder)


def detect_scan(
    model: YOLO,
    scan_name: str,
    image_paths: list[Path],
    config: DetectConfig,
) -> tuple[list[DetectionRow], list[np.ndarray]]:
    """Run the detector. Crops are returned in the same order as rows that need OCR."""
    n_real = len(image_paths)
    source = pad_to_batch([str(p) for p in image_paths], config.detector_batch)
    if len(source) != n_real:
        print(f"  padding {len(source) - n_real} image(s) so the detector batch of {config.detector_batch} is full")
    predict_kwargs: dict = {
        "source": source,
        "imgsz": config.imgsz,
        "conf": config.conf,
        "iou": config.iou,
        "batch": config.detector_batch,
        "stream": True,
        "verbose": False,
    }
    if config.device != "auto":
        predict_kwargs["device"] = config.device

    rows: list[DetectionRow] = []
    crops: list[np.ndarray] = []
    path_by_name = {p.name: p for p in image_paths}
    seen: set[str] = set()

    for index, result in enumerate(model.predict(**predict_kwargs)):
        if index >= n_real:
            break
        image_name = Path(result.path).name
        seen.add(image_name)
        image_bgr = result.orig_img
        boxes = result.boxes
        kept = 0
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            for box_index, ((x1, y1, x2, y2), score, class_id) in enumerate(zip(xyxy, scores, class_ids, strict=True)):
                class_name = model.names.get(int(class_id), str(int(class_id)))
                if config.class_filter is not None and class_name not in config.class_filter:
                    continue
                crop = crop_box(image_bgr, float(x1), float(y1), float(x2), float(y2))
                if crop is None:
                    continue
                if config.save_crops_dir is not None:
                    scan_dir = config.save_crops_dir / scan_name
                    scan_dir.mkdir(parents=True, exist_ok=True)
                    out_path = scan_dir / f"{Path(image_name).stem}_{box_index}_{class_name}.jpg"
                    cv2.imwrite(str(out_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                crops.append(crop)
                rows.append(
                    DetectionRow(
                        scan=scan_name,
                        image=image_name,
                        class_name=class_name,
                        det_conf=float(score),
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        plate="",
                        char_conf=None,
                        region=None,
                        region_prob=None,
                    )
                )
                kept += 1
        if kept == 0:
            rows.append(empty_row(scan_name, image_name))

    missing = [name for name in path_by_name if name not in seen]
    rows.extend(empty_row(scan_name, image_name) for image_name in missing)
    return rows, crops


def attach_ocr(rows: list[DetectionRow], predictions: list) -> None:
    pending = [row for row in rows if row.det_conf is not None]
    if len(pending) != len(predictions):
        raise RuntimeError(f"OCR returned {len(predictions)} predictions for {len(pending)} crops.")
    for row, pred in zip(pending, predictions, strict=True):
        row.plate = pred.plate or ""
        row.char_conf = mean_char_conf(pred.char_probs)
        row.region = pred.region
        row.region_prob = float(pred.region_prob) if pred.region_prob is not None else None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def vote_scan(rows: list[DetectionRow], vote_class: str) -> dict[str, str | int | float | None]:
    """Majority plate for one inspection. Prefer the vote class, then any other class."""
    images = {row.image for row in rows}
    detections = [row for row in rows if row.det_conf is not None]
    main_reads = [row for row in detections if row.plate and row.class_name == vote_class]
    any_reads = [row for row in detections if row.plate]
    if main_reads:
        pool = main_reads
        vote_source = vote_class
    else:
        pool = any_reads
        vote_source = "all" if any_reads else ""

    grouped: dict[str, list[DetectionRow]] = defaultdict(list)
    for row in pool:
        grouped[row.plate].append(row)

    plate = ""
    votes = 0
    mean_char = None
    mean_det = None
    if grouped:

        def rank(text: str) -> tuple[int, float, float]:
            group = grouped[text]
            return (
                len(group),
                _mean([row.char_conf for row in group if row.char_conf is not None]) or 0.0,
                _mean([row.det_conf for row in group if row.det_conf is not None]) or 0.0,
            )

        plate = max(grouped, key=rank)
        chosen = grouped[plate]
        votes = len(chosen)
        mean_char = _mean([row.char_conf for row in chosen if row.char_conf is not None])
        mean_det = _mean([row.det_conf for row in chosen if row.det_conf is not None])

    alternatives = "; ".join(
        f"{text}:{len(group)}" for text, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
    )
    return {
        "scan": rows[0].scan if rows else "",
        "n_images": len(images),
        "n_detections": len(detections),
        "n_reads": len(any_reads),
        "vote_source": vote_source,
        "plate": plate,
        "votes": votes,
        "mean_char_conf": mean_char,
        "mean_det_conf": mean_det,
        "alternatives": alternatives,
    }


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def write_detections(path: Path, rows: list[DetectionRow]) -> None:
    fieldnames = [
        "scan",
        "image",
        "class_name",
        "det_conf",
        "x1",
        "y1",
        "x2",
        "y2",
        "plate",
        "char_conf",
        "region",
        "region_prob",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scan": row.scan,
                    "image": row.image,
                    "class_name": row.class_name,
                    "det_conf": fmt(row.det_conf),
                    "x1": fmt(row.x1),
                    "y1": fmt(row.y1),
                    "x2": fmt(row.x2),
                    "y2": fmt(row.y2),
                    "plate": row.plate,
                    "char_conf": fmt(row.char_conf),
                    "region": row.region or "",
                    "region_prob": fmt(row.region_prob),
                }
            )


def write_scans(path: Path, summaries: list[dict]) -> None:
    fieldnames = [
        "scan",
        "n_images",
        "n_detections",
        "n_reads",
        "vote_source",
        "plate",
        "votes",
        "mean_char_conf",
        "mean_det_conf",
        "alternatives",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    **summary,
                    "mean_char_conf": fmt(summary["mean_char_conf"]),  # type: ignore[arg-type]
                    "mean_det_conf": fmt(summary["mean_det_conf"]),  # type: ignore[arg-type]
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "scans_root",
        nargs="?",
        type=Path,
        default=SCANS_ROOT,
        help="Directory of inspection folders. Each subfolder is one inspection (default: SCANS_ROOT).",
    )
    parser.add_argument("--detector", type=Path, default=DETECTOR, help="Ultralytics .pt checkpoint.")
    parser.add_argument("--ocr-model", default=OCR_MODEL, help="fast-plate-ocr hub model name.")
    parser.add_argument("--conf", type=float, default=CONF, help="Detector confidence threshold.")
    parser.add_argument("--iou", type=float, default=IOU, help="Detector NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=IMGSZ, help="Detector inference size.")
    parser.add_argument(
        "--detector-batch",
        type=int,
        default=DETECTOR_BATCH,
        help="Fixed detector batch size. Short batches are padded by repeating the last image.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=CLASSES,
        help="Detector class names to keep. Default: CLASSES, or every class when that is None.",
    )
    parser.add_argument(
        "--vote-class",
        default=VOTE_CLASS,
        help="Class used for the per-inspection plate vote. Falls back to every class if this one has no reads.",
    )
    parser.add_argument("--device", default=DEVICE, help="Ultralytics device: auto, cpu, mps, or 0.")
    parser.add_argument("--ocr-device", default=OCR_DEVICE, choices=("auto", "cpu", "cuda"), help="OCR device.")
    parser.add_argument("--ocr-batch", type=int, default=OCR_BATCH, help="Number of crops per OCR batch.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Directory for detections.csv and scans.csv.")
    parser.add_argument(
        "--save-crops",
        action=argparse.BooleanOptionalAction,
        default=SAVE_CROPS,
        help="Write plate crops under the output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scans_root = args.scans_root.expanduser().resolve()
    detector_path = args.detector.expanduser().resolve()
    if not detector_path.is_file():
        print(f"Error: detector checkpoint not found: {detector_path}", file=sys.stderr)
        return 1

    try:
        scans = discover_scans(scans_root)
    except NotADirectoryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not scans:
        print(f"No scan folders with images found in {scans_root}", file=sys.stderr)
        return 1

    print(f"Loading detector {detector_path.name}...")
    model = YOLO(str(detector_path))
    known_classes = set(model.names.values())
    class_filter = set(args.classes) if args.classes else None
    if class_filter is not None:
        unknown = class_filter - known_classes
        if unknown:
            print(
                f"Error: unknown class names {sorted(unknown)}. Model classes: {sorted(known_classes)}",
                file=sys.stderr,
            )
            return 1

    print(f"Loading OCR model '{args.ocr_model}'...")
    recognizer = LicensePlateRecognizer(hub_ocr_model=args.ocr_model, device=args.ocr_device)

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops" if args.save_crops else None

    all_rows: list[DetectionRow] = []
    summaries: list[dict] = []
    print(f"Processing {len(scans)} inspection(s) from {scans_root}\n")

    for scan_name, scan_dir in scans:
        image_paths = collect_images(scan_dir)
        print(f"{scan_name}: {len(image_paths)} image(s)")
        rows, crops = detect_scan(
            model,
            scan_name,
            image_paths,
            DetectConfig(
                class_filter=class_filter,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                detector_batch=args.detector_batch,
                save_crops_dir=crops_dir,
            ),
        )
        if crops:
            attach_ocr(rows, read_crops(recognizer, crops, args.ocr_batch))
        summary = vote_scan(rows, args.vote_class)
        summaries.append(summary)
        all_rows.extend(rows)
        plate = summary["plate"] or "(no read)"
        print(
            f"  detections={summary['n_detections']} reads={summary['n_reads']} "
            f"plate={plate} votes={summary['votes']}/{summary['n_reads']}"
        )

    detections_path = output_dir / "detections.csv"
    scans_path = output_dir / "scans.csv"
    write_detections(detections_path, all_rows)
    write_scans(scans_path, summaries)
    print(f"\nWrote {detections_path}")
    print(f"Wrote {scans_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
