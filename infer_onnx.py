#!/usr/bin/env python3
"""
Minimal ONNX inference for the cct_s_v2_global plate OCR model.

Expects an NCHW model exported with:
    python export_onnx_and_trt.py \
        --keras-model cct_s_v2_global.keras \
        --plate-config cct_s_v2_global_plate_config.yaml \
        --data-format channels_first

No dependency on the fast_plate_ocr library — only onnxruntime, opencv, numpy, pyyaml.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

# ── Defaults (cct_s_v2_global) ──────────────────────────────────────────────
DEFAULT_ONNX = Path(__file__).parent / "cct_s_v2_global.onnx"
DEFAULT_CONFIG = Path(__file__).parent / "cct_s_v2_global_plate_config.yaml"
DEFAULT_FOLDER = Path(
    "/Users/anwesh.marwade@pon.com/repos/ai-library/src/ai_library/assets/dekra_images_debug/dekra_images_crop"
)


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def preprocess_single(img_bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    """BGR image → (H, W, 3) uint8 RGB array."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def build_batch_nchw(frames: list[np.ndarray]) -> np.ndarray:
    """Stack (H, W, C) frames into a single (N, C, H, W) uint8 batch."""
    nhwc = np.stack(frames, axis=0)  # (N, H, W, C)
    return np.ascontiguousarray(nhwc.transpose(0, 3, 1, 2))  # (N, C, H, W)


def decode_plates(
    plate_output: np.ndarray, alphabet: str, max_slots: int, pad_char: str
) -> list[tuple[str, float]]:
    """Decode raw plate tensor → list of (plate_text, avg_confidence)."""
    preds = plate_output.reshape(-1, max_slots, len(alphabet))
    indices = np.argmax(preds, axis=-1)
    confidences = np.max(preds, axis=-1)
    chars = np.array(list(alphabet))
    results = []
    for row, conf_row in zip(indices, confidences):
        plate = "".join(chars[row]).rstrip(pad_char)
        avg_conf = float(conf_row[: len(plate)].mean()) if plate else 0.0
        results.append((plate, avg_conf))
    return results


def decode_regions(region_output: np.ndarray, region_labels: list[str]) -> list[tuple[str, float]]:
    """Decode raw region tensor → list of (region_name, probability)."""
    indices = np.argmax(region_output, axis=-1)
    probs = region_output[np.arange(len(indices)), indices]
    return [(region_labels[i], float(p)) for i, p in zip(indices, probs)]


def detect_layout(sess: ort.InferenceSession) -> str:
    """Return 'nchw' or 'nhwc' based on the ONNX model's input shape."""
    shape = sess.get_inputs()[0].shape  # e.g. ['N', 3, 64, 128] or ['N', 64, 128, 3]
    # The channel dim (small value like 1 or 3) is at index 1 for NCHW, index 3 for NHWC.
    # Use the static dims to decide.
    static = [d for d in shape[1:] if isinstance(d, int)]
    if len(static) >= 3 and static[0] <= 4:
        return "nchw"
    return "nhwc"


def main() -> int:
    p = argparse.ArgumentParser(description="Run ONNX plate OCR on a folder of images (batched, NCHW).")
    p.add_argument("folder", nargs="?", type=Path, default=DEFAULT_FOLDER, help="Folder of plate crop images.")
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Path to .onnx model.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to plate_config.yaml.")
    p.add_argument("--batch-size", type=int, default=0, help="Batch size (0 = all images in one batch).")
    args = p.parse_args()

    cfg = load_config(args.config)
    alphabet = cfg["alphabet"]
    max_slots = cfg["max_plate_slots"]
    pad_char = cfg["pad_char"]
    h, w = cfg["img_height"], cfg["img_width"]
    regions = cfg.get("plate_regions")

    sess = ort.InferenceSession(str(args.onnx), providers=ort.get_available_providers())
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    has_region = "region" in output_names and regions
    layout = detect_layout(sess)

    image_paths = sorted(
        ip for ip in args.folder.iterdir() if ip.is_file() and ip.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"No images found in {args.folder}")
        return 1

    # Read and preprocess all images
    valid_paths: list[Path] = []
    frames: list[np.ndarray] = []
    for ip in image_paths:
        img_bgr = cv2.imread(str(ip))
        if img_bgr is None:
            print(f"{ip.name}\t[read error]")
            continue
        valid_paths.append(ip)
        frames.append(preprocess_single(img_bgr, h, w))

    if not frames:
        print("No valid images to process.")
        return 1

    batch_size = args.batch_size if args.batch_size > 0 else len(frames)
    all_plates: list[tuple[str, float]] = []
    all_regions: list[tuple[str, float]] = []

    for start in range(0, len(frames), batch_size):
        chunk = frames[start : start + batch_size]
        if layout == "nchw":
            tensor = build_batch_nchw(chunk)
        else:
            tensor = np.stack(chunk, axis=0).astype(np.uint8)

        outs = sess.run(output_names, {input_name: tensor})
        out_map = dict(zip(output_names, outs))

        all_plates.extend(decode_plates(out_map["plate"], alphabet, max_slots, pad_char))
        if has_region:
            all_regions.extend(decode_regions(out_map["region"], regions))

    print(f"Processed {len(valid_paths)} image(s)  [layout={layout}, batch_size={batch_size}]\n")
    for i, ip in enumerate(valid_paths):
        plate_text, conf = all_plates[i]
        line = f"{ip.name}\t{plate_text}\tconf={conf:.2f}"
        if has_region:
            region_name, region_prob = all_regions[i]
            line += f"\tregion={region_name} ({region_prob:.2f})"
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
