#!/usr/bin/env python3
"""
Minimal TensorRT inference for the cct_s_v2_global plate OCR model.

Uses PyTorch for GPU memory management (no pycuda / cuda-python needed).

Build the engine first:
    python export_onnx_and_trt.py trt --onnx cct_s_v2_global.onnx --trt-fp16

Only needs: tensorrt, torch, opencv, numpy, pyyaml.
No dependency on fast_plate_ocr.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}

DEFAULT_ENGINE = Path(__file__).parent / "cct_s_v2_global.engine"
DEFAULT_CONFIG = Path(__file__).parent / "cct_s_v2_global_plate_config.yaml"
DEFAULT_FOLDER = Path(
    "/Users/anwesh.marwade@pon.com/repos/ai-library/src/ai_library/assets/dekra_images_debug/dekra_images_crop"
)

TRT_TO_TORCH_DTYPE = {
    0: torch.float32,   # trt.float32
    1: torch.float16,   # trt.float16
    2: torch.int8,      # trt.int8
    3: torch.int32,     # trt.int32
    4: torch.bool,      # trt.bool
    6: torch.float16,   # trt.bf16 → closest common fallback
}


# ── TensorRT runtime wrapper ────────────────────────────────────────────────


class TRTRunner:
    """Thin wrapper around a deserialized TensorRT engine using PyTorch for CUDA memory."""

    def __init__(self, engine_path: Path, device: str = "cuda:0"):
        import tensorrt as trt

        self._trt = trt
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(self.device)

        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names: list[str] = []
        self.output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Run synchronous inference.

        Args:
            inputs: mapping of input tensor name → contiguous numpy array.

        Returns:
            mapping of output tensor name → numpy array with results.
        """
        with torch.cuda.stream(self.stream):
            # Upload inputs to GPU
            d_inputs: dict[str, torch.Tensor] = {}
            for name, arr in inputs.items():
                self.context.set_input_shape(name, arr.shape)
                t = torch.from_numpy(arr).to(self.device, non_blocking=True)
                d_inputs[name] = t.contiguous()
                self.context.set_tensor_address(name, d_inputs[name].data_ptr())

            # Allocate output tensors on GPU
            d_outputs: dict[str, torch.Tensor] = {}
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                trt_dtype = self.engine.get_tensor_dtype(name)
                torch_dtype = TRT_TO_TORCH_DTYPE.get(int(trt_dtype), torch.float32)
                d_outputs[name] = torch.empty(shape, dtype=torch_dtype, device=self.device)
                self.context.set_tensor_address(name, d_outputs[name].data_ptr())

            # Execute
            self.context.execute_async_v3(self.stream.cuda_stream)

        self.stream.synchronize()

        return {name: t.cpu().numpy() for name, t in d_outputs.items()}


# ── Preprocessing / postprocessing (same as infer_onnx.py) ──────────────────


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def preprocess_single(img_bgr: np.ndarray, h: int, w: int) -> np.ndarray:
    """BGR image → (H, W, 3) uint8 RGB array."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def build_batch_nhwc(frames: list[np.ndarray]) -> np.ndarray:
    """Stack (H, W, C) frames into (N, H, W, C) uint8."""
    return np.ascontiguousarray(np.stack(frames, axis=0).astype(np.uint8))


def build_batch_nchw(frames: list[np.ndarray]) -> np.ndarray:
    """Stack (H, W, C) frames into (N, C, H, W) uint8."""
    nhwc = np.stack(frames, axis=0)
    return np.ascontiguousarray(nhwc.transpose(0, 3, 1, 2).astype(np.uint8))


def detect_layout(engine_input_shape: tuple[int, ...]) -> str:
    """Return 'nchw' or 'nhwc' from the engine's input shape (ignoring batch)."""
    spatial = engine_input_shape[1:]  # drop batch dim
    if len(spatial) >= 3 and spatial[0] <= 4:
        return "nchw"
    return "nhwc"


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


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="Run TensorRT plate OCR on a folder of images (batched).")
    p.add_argument("folder", nargs="?", type=Path, default=DEFAULT_FOLDER, help="Folder of plate crop images.")
    p.add_argument("--engine", type=Path, default=DEFAULT_ENGINE, help="Path to .engine file.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to plate_config.yaml.")
    p.add_argument("--batch-size", type=int, default=0, help="Batch size (0 = all images in one batch).")
    args = p.parse_args()

    cfg = load_config(args.config)
    alphabet = cfg["alphabet"]
    max_slots = cfg["max_plate_slots"]
    pad_char = cfg["pad_char"]
    h, w = cfg["img_height"], cfg["img_width"]
    regions = cfg.get("plate_regions")

    runner = TRTRunner(args.engine)

    input_name = runner.input_names[0]
    has_region = "region" in runner.output_names and regions

    # Detect layout from the engine's input shape (use opt profile shape)
    engine_shape = runner.engine.get_tensor_shape(input_name)
    layout = detect_layout(tuple(engine_shape))

    image_paths = sorted(
        ip for ip in args.folder.iterdir() if ip.is_file() and ip.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"No images found in {args.folder}")
        return 1

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
        tensor = build_batch_nchw(chunk) if layout == "nchw" else build_batch_nhwc(chunk)

        out_map = runner.infer({input_name: tensor})

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
