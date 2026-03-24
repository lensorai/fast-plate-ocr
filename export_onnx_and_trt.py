#!/usr/bin/env python3
"""
Export a fast-plate-ocr Keras model to ONNX and optionally build a TensorRT engine.

Usage examples
--------------
# ONNX only (channels-last, uint8 — matches default inference pipeline):
python export_onnx_and_trt.py \
    --keras-model cct_s_v2_global.keras \
    --plate-config config/latin_plate_config_v2.yaml \
    --data-format channels_first \
    --input-dtype float32

# ONNX + TensorRT engine (FP16, NCHW for optimal TRT performance):
python export_onnx_and_trt.py \
    --keras-model cct_s_v2_global.keras \
    --plate-config config/latin_plate_config_v2.yaml \
    --data-format channels_first \
    --input-dtype float32
    --build-trt \
    --trt-fp16

# ONNX + TensorRT with custom batch profile:
python export_onnx_and_trt.py \
    --keras-model cct_s_v2_global.keras \
    --plate-config cct_s_v2_global_plate_config.yaml \
    --build-trt \
    --trt-fp16 \
    --trt-min-batch 1 \
    --trt-opt-batch 8 \
    --trt-max-batch 32
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import shutil
import sys
from tempfile import NamedTemporaryFile

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _load_keras_model(keras_path: pathlib.Path, plate_config_path: pathlib.Path):
    """Load a .keras checkpoint with all custom objects registered."""
    import keras

    from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
    from fast_plate_ocr.train.utilities.utils import load_keras_model

    plate_cfg = load_plate_config_from_yaml(plate_config_path)
    model = load_keras_model(keras_path, plate_cfg)
    return model, plate_cfg


def export_to_onnx(
    keras_path: pathlib.Path,
    plate_config_path: pathlib.Path,
    out_onnx: pathlib.Path,
    *,
    input_dtype: str = "uint8",
    data_format: str = "channels_last",
    dynamic_batch: bool = True,
    simplify: bool = True,
    opset_version: int | None = None,
) -> pathlib.Path:
    """Export a Keras model to ONNX, validate, and return the output path."""
    import keras
    import onnxruntime as ort

    model, plate_cfg = _load_keras_model(keras_path, plate_config_path)

    if data_format == "channels_first":
        inp_shape = (plate_cfg.num_channels, plate_cfg.img_height, plate_cfg.img_width)
        x_in = keras.Input(shape=inp_shape, dtype=input_dtype, name="input_nchw")
        x_out = model(keras.layers.Permute((2, 3, 1))(x_in))
        export_model = keras.Model(x_in, x_out, name=f"{model.name}_nchw")
    else:
        inp_shape = (plate_cfg.img_height, plate_cfg.img_width, plate_cfg.num_channels)
        export_model = model

    batch_dim = None if dynamic_batch else 1
    spec_shape = (batch_dim, *inp_shape)
    spec = [keras.InputSpec(name="input", shape=spec_shape, dtype=input_dtype)]

    log.info("Exporting Keras → ONNX  (shape=%s, dtype=%s) ...", spec_shape, input_dtype)

    with NamedTemporaryFile(suffix=".onnx") as tmp:
        export_model.export(
            tmp.name,
            format="onnx",
            verbose=False,
            input_signature=spec,
            opset_version=opset_version,
        )

        if simplify:
            import onnx
            import onnxslim

            log.info("Simplifying ONNX graph with onnxslim ...")
            simplified = onnxslim.slim(onnx.load(tmp.name))
            onnx.save(simplified, str(out_onnx))
        else:
            shutil.copy(tmp.name, out_onnx)

    # Validate round-trip
    sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    dummy = np.random.randint(0, 256, size=(1, *inp_shape)).astype(input_dtype)
    onnx_out = sess.run(None, {"input": dummy})
    keras_out = export_model.predict(dummy, verbose=0)

    if isinstance(keras_out, dict):
        keras_values = list(keras_out.values())
    elif isinstance(keras_out, (list, tuple)):
        keras_values = list(keras_out)
    else:
        keras_values = [keras_out]

    all_close = True
    for i, (k_val, o_val) in enumerate(zip(keras_values, onnx_out, strict=False)):
        if np.allclose(k_val, o_val, rtol=1e-4, atol=1e-4):
            log.info("Output %d: Keras ↔ ONNX match ✔", i)
        else:
            log.warning("Output %d: Keras ↔ ONNX MISMATCH (max Δ=%.6f)", i, np.max(np.abs(k_val - o_val)))
            all_close = False

    if all_close:
        log.info("ONNX export validated successfully.")
    else:
        log.warning("ONNX outputs deviate from Keras — check tolerances.")

    log.info("Saved ONNX model → %s", out_onnx)
    return out_onnx


def build_trt_engine(
    onnx_path: pathlib.Path,
    engine_path: pathlib.Path,
    *,
    fp16: bool = True,
    int8: bool = False,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 16,
) -> pathlib.Path:
    """
    Build a TensorRT engine from an ONNX model.

    Requires the ``tensorrt`` Python package (pip install tensorrt).
    """
    try:
        import tensorrt as trt
    except ImportError:
        log.error(
            "tensorrt package not found. Install it with:\n"
            "  pip install tensorrt\n"
            "Or use trtexec CLI instead:\n"
            "  trtexec --onnx=%s --saveEngine=%s %s",
            onnx_path,
            engine_path,
            "--fp16" if fp16 else "",
        )
        sys.exit(1)

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    log.info("Parsing ONNX model: %s", onnx_path)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error("TRT ONNX parse error: %s", parser.get_error(i))
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GiB

    if fp16:
        if not builder.platform_has_fast_fp16:
            log.warning("Platform does not have fast FP16 — enabling anyway.")
        config.set_flag(trt.BuilderFlag.FP16)
        log.info("FP16 enabled.")
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        log.info("INT8 enabled (you may need a calibrator for best accuracy).")

    # Build optimization profile for the dynamic batch dimension
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = input_tensor.shape  # e.g. (-1, 64, 128, 3)
    spatial_dims = tuple(input_shape[1:])

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        min=(min_batch, *spatial_dims),
        opt=(opt_batch, *spatial_dims),
        max=(max_batch, *spatial_dims),
    )
    config.add_optimization_profile(profile)
    log.info(
        "TRT profile: input=%s  min_batch=%d  opt_batch=%d  max_batch=%d",
        input_name,
        min_batch,
        opt_batch,
        max_batch,
    )

    log.info("Building TensorRT engine (this may take a few minutes) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        log.error("TensorRT engine build failed.")
        sys.exit(1)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    log.info("Saved TensorRT engine → %s", engine_path)
    return engine_path


def print_trtexec_command(
    onnx_path: pathlib.Path,
    engine_path: pathlib.Path,
    *,
    fp16: bool,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    input_name: str = "input",
    spatial: str = "64x128x3",
) -> None:
    """Print the equivalent trtexec command for reference."""
    flags = ["--fp16"] if fp16 else []
    cmd = " \\\n    ".join(
        [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            *flags,
            f"--minShapes={input_name}:{min_batch}x{spatial}",
            f"--optShapes={input_name}:{opt_batch}x{spatial}",
            f"--maxShapes={input_name}:{max_batch}x{spatial}",
        ]
    )
    log.info("Equivalent trtexec command:\n\n  %s\n", cmd)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export fast-plate-ocr Keras model to ONNX and optionally TensorRT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Keras / ONNX args ---
    p.add_argument("--keras-model", required=True, type=pathlib.Path, help="Path to .keras model file.")
    p.add_argument("--plate-config", required=True, type=pathlib.Path, help="Path to plate_config.yaml.")
    p.add_argument("--out-dir", type=pathlib.Path, default=None, help="Output directory (default: same as --keras-model).")
    p.add_argument("--input-dtype", choices=["uint8", "float32"], default="uint8", help="ONNX input dtype (default: uint8).")
    p.add_argument(
        "--data-format",
        choices=["channels_last", "channels_first"],
        default="channels_last",
        help="Input tensor layout. channels_first (NCHW) is preferred for TensorRT (default: channels_last).",
    )
    p.add_argument("--no-simplify", action="store_true", help="Skip onnxslim graph simplification.")
    p.add_argument("--no-dynamic-batch", action="store_true", help="Use static batch=1 instead of dynamic.")
    p.add_argument("--opset", type=int, default=None, help="ONNX opset version (default: Keras default).")

    # --- TensorRT args ---
    p.add_argument("--build-trt", action="store_true", help="Also build a TensorRT .engine file from the ONNX model.")
    p.add_argument("--trt-fp16", action="store_true", help="Enable FP16 precision for TRT engine.")
    p.add_argument("--trt-int8", action="store_true", help="Enable INT8 precision for TRT engine (needs calibrator for best results).")
    p.add_argument("--trt-min-batch", type=int, default=1, help="TRT profile: minimum batch size (default: 1).")
    p.add_argument("--trt-opt-batch", type=int, default=1, help="TRT profile: optimal batch size (default: 1).")
    p.add_argument("--trt-max-batch", type=int, default=16, help="TRT profile: maximum batch size (default: 16).")

    args = p.parse_args()

    keras_path: pathlib.Path = args.keras_model.resolve()
    config_path: pathlib.Path = args.plate_config.resolve()
    out_dir = (args.out_dir or keras_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = keras_path.stem
    onnx_path = out_dir / f"{stem}.onnx"
    engine_path = out_dir / f"{stem}.engine"

    # ── Step 1: Keras → ONNX ──
    export_to_onnx(
        keras_path,
        config_path,
        onnx_path,
        input_dtype=args.input_dtype,
        data_format=args.data_format,
        dynamic_batch=not args.no_dynamic_batch,
        simplify=not args.no_simplify,
        opset_version=args.opset,
    )

    # ── Step 2 (optional): ONNX → TensorRT ──
    if args.build_trt:
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        h, w = cfg["img_height"], cfg["img_width"]
        c = 3 if cfg.get("image_color_mode", "grayscale") == "rgb" else 1
        if args.data_format == "channels_first":
            spatial = f"{c}x{h}x{w}"
        else:
            spatial = f"{h}x{w}x{c}"

        print_trtexec_command(
            onnx_path,
            engine_path,
            fp16=args.trt_fp16,
            min_batch=args.trt_min_batch,
            opt_batch=args.trt_opt_batch,
            max_batch=args.trt_max_batch,
            spatial=spatial,
        )

        build_trt_engine(
            onnx_path,
            engine_path,
            fp16=args.trt_fp16,
            int8=args.trt_int8,
            min_batch=args.trt_min_batch,
            opt_batch=args.trt_opt_batch,
            max_batch=args.trt_max_batch,
        )

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
