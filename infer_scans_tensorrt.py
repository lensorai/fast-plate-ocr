#!/usr/bin/env python3
"""Run the vision-core-2 license-plate TensorRT pipeline on scan folders.

This is a standalone reproduction of ``LicensePlateStream``:

1. letterbox each source image to the detector input using RGB 114 padding;
2. run a detector TensorRT engine with NMS already embedded;
3. keep every reported detection, but OCR only the highest-confidence class-0
   (``license_plate_main``) detection for each image;
4. crop the original RGB image with truncated integer coordinates and stretch
   it bilinearly to the OCR input shape, without crop padding or letterboxing;
5. decode either fast-plate-ocr CCT outputs (``plate``/``region``) or legacy
   YOLO-character outputs. Dutch and Belgian CCT reads are hyphen formatted.

The detector and OCR engines must have been built for the current NVIDIA GPU
and TensorRT version. The detector is expected to include NMS, like the
vision-core-2 engine; this script deliberately does not run a second NMS.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
GRAY_COLOR = 114
LP_MAIN_CLASS_ID = 0

# Edit these. Matching command-line flags override them.
SCANS_ROOT = Path("/Users/anwesh.marwade@pon.com/Downloads/distrifresh_26_10")
DETECTOR_ENGINE = Path(
    "/Users/anwesh.marwade@pon.com/Downloads/lux_vision_core_models_converted/"
    "license_plate/2026-02-20-3-LUX/yolov8_mask_640/weights/model.engine"
)
DETECTOR_CONFIG = Path(
    "/Users/anwesh.marwade@pon.com/Downloads/lux_vision_core_models_converted/"
    "license_plate/2026-02-20-3-LUX/model_configuration.json"
)
OCR_ENGINE = Path("/Users/anwesh.marwade@pon.com/Downloads/2026-10-06_1-LUX-3060/cctsv2g_128/weights/model.engine")
OCR_CONFIG = Path("/Users/anwesh.marwade@pon.com/Downloads/fast-ocr-model-configuration_lux.json")
DEVICE_ID = 0
OUTPUT_DIR = Path("scan_ocr_tensorrt_out")
SAVE_CROPS = True

PLATE_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
PLATE_PAD_CHAR = "_"
PLATE_REGIONS: tuple[str, ...] = (
    "Albania",
    "Andorra",
    "Argentina",
    "Armenia",
    "Australia",
    "Austria",
    "Azerbaijan",
    "Bahrain",
    "Belarus",
    "Belgium",
    "Bosnia and Herzegovina",
    "Brazil",
    "Bulgaria",
    "Cambodia",
    "Canada",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Georgia",
    "Germany",
    "Gibraltar",
    "Greece",
    "Guernsey",
    "Hungary",
    "Iceland",
    "Indonesia",
    "Ireland",
    "Israel",
    "Italy",
    "Latvia",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Malaysia",
    "Malta",
    "Mexico",
    "Moldova",
    "Monaco",
    "Montenegro",
    "Netherlands",
    "New Zealand",
    "North Macedonia",
    "Norway",
    "Poland",
    "Portugal",
    "Qatar",
    "Romania",
    "San Marino",
    "Serbia",
    "Singapore",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "Switzerland",
    "Thailand",
    "Turkey",
    "United States",
    "Ukraine",
    "United Kingdom",
    "Vietnam",
    "Unknown",
)
DUTCH_LAYOUTS = {
    "DLLLDD": (1, 4),
    "LLDDDL": (2, 5),
    "LDDDLL": (1, 4),
    "LLLDDL": (3, 5),
    "LLDDLL": (2, 4),
    "LLLLDD": (2, 4),
    "DDLLLL": (2, 4),
    "DDLLLD": (2, 5),
    "LDDLLL": (1, 3),
}
BELGIAN_LAYOUTS = {"DLLLDDD": (1, 4), "LLLLDDD": (1, 4)}


@dataclass(frozen=True)
class ModelConfig:
    height: int
    width: int
    batch_size: int
    confidence: float
    labels: dict[int, str]


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
    raw_plate: str = ""
    plate: str = ""
    char_conf: float | None = None
    region: str | None = None
    region_prob: float | None = None


def load_model_config(path: Path) -> ModelConfig:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    parameters = data["parameters"]
    labels = {int(label): name for label, name in data.get("label_mapping", {}).items()}
    return ModelConfig(
        height=int(data["input_size"]["height"]),
        width=int(data["input_size"]["width"]),
        batch_size=int(parameters["backend"]["tensorrt"]["batch_size"]),
        confidence=float(parameters["confidence_threshold"]),
        labels=labels,
    )


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name,
    )


def discover_scans(root: Path) -> list[tuple[str, Path]]:
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    scans = [
        (path.name, path)
        for path in sorted(root.iterdir(), key=lambda candidate: candidate.name)
        if path.is_dir() and not path.name.startswith(".") and collect_images(path)
    ]
    return scans or ([(root.name, root)] if collect_images(root) else [])


def format_plate(raw: str, region: str) -> str:
    layouts = DUTCH_LAYOUTS if region == "Netherlands" else BELGIAN_LAYOUTS if region == "Belgium" else None
    if layouts is None:
        return raw
    shape = "".join("L" if char.isalpha() else "D" if char.isdigit() else "?" for char in raw)
    layout = layouts.get(shape)
    if layout is None:
        return raw
    first, second = layout
    return f"{raw[:first]}-{raw[first:second]}-{raw[second:]}"


def letterbox_rgb(image_rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """DALI-equivalent centered RGB-114 letterbox for detector preprocessing."""
    image_height, image_width = image_rgb.shape[:2]
    gain = min(width / image_width, height / image_height)
    resized_width = round(image_width * gain)
    resized_height = round(image_height * gain)
    if (resized_width, resized_height) != (image_width, image_height):
        image_rgb = cv2.resize(image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    left = round((width - resized_width) / 2 - 0.1)
    right = round((width - resized_width) / 2 + 0.1)
    top = round((height - resized_height) / 2 - 0.1)
    bottom = round((height - resized_height) / 2 + 0.1)
    return cv2.copyMakeBorder(
        image_rgb,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(GRAY_COLOR, GRAY_COLOR, GRAY_COLOR),
    )


def scale_boxes(boxes: torch.Tensor, input_height: int, input_width: int, original_shape: tuple[int, int]) -> None:
    """In-place copy of vision-core-2's padding-aware ``scale_boxes`` path."""
    original_height, original_width = original_shape
    gain = min(input_height / original_height, input_width / original_width)
    pad_x = round((input_width - original_width * gain) / 2 - 0.1)
    pad_y = round((input_height - original_height * gain) / 2 - 0.1)
    boxes[..., (0, 2)] -= pad_x
    boxes[..., (1, 3)] -= pad_y
    boxes[..., :4] /= gain
    boxes[..., (0, 2)].clamp_(0, original_width)
    boxes[..., (1, 3)].clamp_(0, original_height)


def trt_dtype_map() -> dict[Any, torch.dtype]:
    import tensorrt as trt  # noqa: PLC0415

    candidates = {
        "FLOAT": torch.float32,
        "HALF": torch.float16,
        "INT8": torch.int8,
        "INT32": torch.int32,
        "INT64": torch.int64,
        "BOOL": torch.bool,
        "BF16": torch.bfloat16,
        "UINT8": torch.uint8,
    }
    return {getattr(trt.DataType, name): dtype for name, dtype in candidates.items() if hasattr(trt.DataType, name)}


class TensorRTRunner:
    """TensorRT engine runner retaining inputs and outputs on the CUDA device."""

    def __init__(self, engine_path: Path, device_id: int) -> None:
        import tensorrt as trt  # noqa: PLC0415

        self.trt = trt
        self.dtype_map = trt_dtype_map()
        self.device = torch.device(f"cuda:{device_id}")
        self.stream = torch.cuda.Stream(self.device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize {engine_path}. Rebuild it with this TensorRT version ({trt.__version__})."
            )
        self.context = self.engine.create_execution_context()
        self.input_names = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
        ]
        if len(self.input_names) != 1:
            raise ValueError(f"{engine_path.name} has {len(self.input_names)} inputs; this script expects exactly one.")

    def infer(self, input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        input_name = self.input_names[0]
        expected_dtype = self.dtype_map[self.engine.get_tensor_dtype(input_name)]
        input_tensor = input_tensor.to(device=self.device, dtype=expected_dtype).contiguous()
        with torch.cuda.stream(self.stream):
            if not self.context.set_input_shape(input_name, tuple(input_tensor.shape)):
                raise ValueError(f"Engine rejected input shape {tuple(input_tensor.shape)} for {input_name}.")
            self.context.set_tensor_address(input_name, input_tensor.data_ptr())
            outputs: dict[str, torch.Tensor] = {}
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(dimension < 0 for dimension in shape):
                    raise ValueError(f"Unresolved dynamic output shape for {name}: {shape}")
                dtype = self.dtype_map[self.engine.get_tensor_dtype(name)]
                outputs[name] = torch.empty(shape, device=self.device, dtype=dtype)
                self.context.set_tensor_address(name, outputs[name].data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed.")
        self.stream.synchronize()
        return outputs


def empty_row(scan: str, image: str) -> DetectionRow:
    return DetectionRow(scan, image, "", None, None, None, None, None)


def decode_cct(
    output: dict[str, torch.Tensor],
    count: int,
    confidence: float,
) -> list[tuple[str, str, float, str, float]]:
    plate_probs = output["plate"][:count]
    region_probs = output["region"][:count]
    if region_probs.shape[-1] != len(PLATE_REGIONS):
        raise ValueError(f"Expected {len(PLATE_REGIONS)} OCR regions, received {region_probs.shape[-1]}.")

    slot_probs, slot_indices = plate_probs.max(dim=-1)
    region_probs_max, region_indices = region_probs.max(dim=-1)
    decoded = []
    for indices, probabilities, region_index, region_probability in zip(
        slot_indices.cpu(),
        slot_probs.cpu(),
        region_indices.cpu(),
        region_probs_max.cpu(),
        strict=True,
    ):
        raw = "".join(PLATE_ALPHABET[index] for index in indices.tolist()).rstrip(PLATE_PAD_CHAR)
        char_conf = float(probabilities[: len(raw)].mean()) if raw else 0.0
        region = PLATE_REGIONS[int(region_index)]
        region_conf = float(region_probability)
        accepted_region = region if region_conf >= confidence else "Unknown"
        plate = format_plate(raw, accepted_region) if raw and char_conf >= confidence else ""
        decoded.append((raw, plate, char_conf, region, region_conf))
    return decoded


def decode_yolo_characters(
    output: dict[str, torch.Tensor],
    count: int,
    confidence: float,
    labels: dict[int, str],
) -> list[tuple[str, str, float, None, None]]:
    predictions = next(iter(output.values()))[:count, :, :6].cpu()
    decoded = []
    for prediction in predictions:
        chars = prediction[prediction[:, 4] > confidence]
        chars = chars[chars[:, 0].argsort()]
        raw = "".join(labels.get(int(character[5]), "") for character in chars).upper()
        char_conf = float(chars[:, 4].mean()) if len(chars) else 0.0
        decoded.append((raw, raw, char_conf, None, None))
    return decoded


def process_scan(  # noqa: PLR0915
    scan_name: str,
    image_paths: list[Path],
    detector: TensorRTRunner,
    detector_config: ModelConfig,
    ocr: TensorRTRunner,
    ocr_config: ModelConfig,
    output_dir: Path,
    save_crops: bool,
) -> list[DetectionRow]:
    """Process a scan with the same detection/selection/crop sequence as LicensePlateStream."""
    rows: list[DetectionRow] = []
    detector_batch = detector_config.batch_size
    for start in range(0, len(image_paths), detector_batch):
        real_paths = image_paths[start : start + detector_batch]
        batch_paths = real_paths + [real_paths[-1]] * (detector_batch - len(real_paths))
        original_images: list[torch.Tensor] = []
        original_shapes: list[tuple[int, int]] = []
        for path in batch_paths:
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Could not read image: {path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            original_images.append(torch.from_numpy(image_rgb).permute(2, 0, 1).contiguous())
            original_shapes.append(image_rgb.shape[:2])

        detector_input = torch.stack(
            [
                torch.from_numpy(
                    letterbox_rgb(
                        image.permute(1, 2, 0).numpy(),
                        detector_config.height,
                        detector_config.width,
                    )
                )
                .permute(2, 0, 1)
                .float()
                / 255.0
                for image in original_images
            ]
        )
        detector_output = next(iter(detector.infer(detector_input).values()))[:, :, :6]

        selected: list[tuple[int, torch.Tensor] | None] = []
        for batch_index, (detections, shape) in enumerate(zip(detector_output, original_shapes, strict=True)):
            valid = detections[detections[:, 4] > detector_config.confidence].clone()
            if valid.numel() == 0:
                selected.append(None)
                continue
            scale_boxes(valid[:, :4], detector_config.height, detector_config.width, shape)
            for box in valid:
                class_id = int(box[5])
                row_path = batch_paths[batch_index]
                rows.append(
                    DetectionRow(
                        scan_name,
                        row_path.name,
                        detector_config.labels.get(class_id, str(class_id)),
                        float(box[4]),
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    )
                )
            main = valid[valid[:, 5] == LP_MAIN_CLASS_ID]
            selected.append((batch_index, main[main[:, 4].argmax()]) if len(main) else None)

        ocr_crops: list[torch.Tensor] = []
        ocr_row_indices: list[int] = []
        for selection in selected[: len(real_paths)]:
            if selection is None:
                continue
            batch_index, box = selection
            x1, y1, x2, y2 = box[:4].int().tolist()
            height, width = original_shapes[batch_index]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = original_images[batch_index][:, y1:y2, x1:x2].float().unsqueeze(0)
            crop = functional.interpolate(
                crop,
                size=(ocr_config.height, ocr_config.width),
                mode="bilinear",
                align_corners=False,
            )
            if save_crops:
                crop_dir = output_dir / "crops" / scan_name
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_bgr = cv2.cvtColor(
                    original_images[batch_index][:, y1:y2, x1:x2].permute(1, 2, 0).numpy(),
                    cv2.COLOR_RGB2BGR,
                )
                cv2.imwrite(str(crop_dir / f"{batch_paths[batch_index].stem}_license_plate_main.jpg"), crop_bgr)
            main_row_index = next(
                index
                for index in range(len(rows) - 1, -1, -1)
                if rows[index].image == batch_paths[batch_index].name and rows[index].class_name == "license_plate_main"
            )
            ocr_crops.append(crop)
            ocr_row_indices.append(main_row_index)

        if ocr_crops:
            ocr_input = torch.cat(ocr_crops, dim=0)
            ocr_output = ocr.infer(ocr_input)
            decoded = (
                decode_cct(ocr_output, len(ocr_crops), ocr_config.confidence)
                if {"plate", "region"} <= ocr_output.keys()
                else decode_yolo_characters(ocr_output, len(ocr_crops), ocr_config.confidence, ocr_config.labels)
            )
            for row_index, (raw, plate, char_conf, region, region_prob) in zip(ocr_row_indices, decoded, strict=True):
                rows[row_index].raw_plate = raw
                rows[row_index].plate = plate
                rows[row_index].char_conf = char_conf
                rows[row_index].region = region
                rows[row_index].region_prob = region_prob

        detected_images = {row.image for row in rows}
        rows.extend(empty_row(scan_name, path.name) for path in real_paths if path.name not in detected_images)
    return rows


def vote_scan(rows: list[DetectionRow]) -> dict[str, str | int | float]:
    detections = [row for row in rows if row.det_conf is not None]
    reads = [row for row in detections if row.plate and row.class_name == "license_plate_main"]
    grouped: dict[str, list[DetectionRow]] = defaultdict(list)
    for row in reads:
        grouped[row.plate].append(row)
    plate = max(
        grouped,
        key=lambda text: (
            len(grouped[text]),
            sum(row.char_conf or 0.0 for row in grouped[text]) / len(grouped[text]),
        ),
        default="",
    )
    return {
        "scan": rows[0].scan,
        "n_images": len({row.image for row in rows}),
        "n_detections": len(detections),
        "n_reads": len(reads),
        "plate": plate,
        "votes": len(grouped[plate]) if plate else 0,
        "alternatives": "; ".join(
            f"{text}:{len(group)}" for text, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
        ),
    }


def write_results(output_dir: Path, rows: list[DetectionRow], summaries: list[dict[str, str | int | float]]) -> None:
    detection_fields = list(DetectionRow.__dataclass_fields__)
    with (output_dir / "detections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detection_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: getattr(row, field) if getattr(row, field) is not None else "" for field in detection_fields}
            )
    with (output_dir / "scans.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]) if summaries else ["scan"])
        writer.writeheader()
        writer.writerows(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scans_root", nargs="?", type=Path, default=SCANS_ROOT)
    parser.add_argument("--detector-engine", type=Path, default=DETECTOR_ENGINE)
    parser.add_argument("--detector-config", type=Path, default=DETECTOR_CONFIG)
    parser.add_argument("--ocr-engine", type=Path, default=OCR_ENGINE)
    parser.add_argument("--ocr-config", type=Path, default=OCR_CONFIG)
    parser.add_argument("--device-id", type=int, default=DEVICE_ID)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--save-crops", action=argparse.BooleanOptionalAction, default=SAVE_CROPS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.detector_engine, args.detector_config, args.ocr_engine, args.ocr_config):
        if not path.is_file():
            print(f"Error: required file not found: {path}", file=sys.stderr)
            return 1
    if not torch.cuda.is_available():
        print("Error: TensorRT requires an NVIDIA CUDA GPU.", file=sys.stderr)
        return 1
    try:
        scans = discover_scans(args.scans_root)
    except NotADirectoryError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if not scans:
        print(f"Error: no scans with images found in {args.scans_root}", file=sys.stderr)
        return 1

    detector_config = load_model_config(args.detector_config)
    ocr_config = load_model_config(args.ocr_config)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = TensorRTRunner(args.detector_engine, args.device_id)
    ocr = TensorRTRunner(args.ocr_engine, args.device_id)

    all_rows: list[DetectionRow] = []
    summaries = []
    for scan_name, scan_dir in scans:
        paths = collect_images(scan_dir)
        print(f"{scan_name}: {len(paths)} image(s)")
        rows = process_scan(
            scan_name,
            paths,
            detector,
            detector_config,
            ocr,
            ocr_config,
            output_dir,
            args.save_crops,
        )
        summary = vote_scan(rows)
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"  detections={summary['n_detections']} reads={summary['n_reads']} plate={summary['plate'] or '(no read)'}"
        )

    write_results(output_dir, all_rows, summaries)
    print(f"Wrote {output_dir / 'detections.csv'}")
    print(f"Wrote {output_dir / 'scans.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
