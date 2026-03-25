#!/usr/bin/env python3
"""
Export a fast-plate-ocr Keras model to ONNX, and/or convert an ONNX model to TensorRT.

Subcommands
-----------
  onnx    Keras (.keras) → ONNX (.onnx)
  trt     ONNX  (.onnx)  → TensorRT (.engine)

Usage examples
--------------
# 1. Keras → ONNX (NHWC uint8, default):
python export_onnx_and_trt.py onnx \
    --keras-model cct_s_v2_global.keras \
    --plate-config cct_s_v2_global_plate_config.yaml

# 2. Keras → ONNX (NCHW float32, for TRT):
python export_onnx_and_trt.py onnx \
    --keras-model cct_s_v2_global.keras \
    --plate-config cct_s_v2_global_plate_config.yaml \
    --data-format channels_first \
    --input-dtype float32

# 3. ONNX → TensorRT (FP16):
python export_onnx_and_trt.py trt \
    --onnx cct_s_v2_global.onnx \
    --trt-fp16

# 4. ONNX → TensorRT (custom batch profile):
python export_onnx_and_trt.py trt \
    --onnx cct_s_v2_global.onnx \
    --trt-fp16 \
    --min-batch 1 --opt-batch 8 --max-batch 32
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Keras → ONNX
# ═══════════════════════════════════════════════════════════════════════════════


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


# ═══════════════════════════════════════════════════════════════════════════════
#  ONNX → TensorRT
# ═══════════════════════════════════════════════════════════════════════════════


def _read_onnx_input_meta(onnx_path: pathlib.Path) -> tuple[str, tuple[int, ...]]:
    """Return (input_name, spatial_dims) by inspecting the ONNX graph."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    inp = model.graph.input[0]
    name = inp.name
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value > 0 else -1)
    return name, tuple(dims[1:])


def build_trt_engine(
    onnx_path: pathlib.Path,
    engine_path: pathlib.Path,
    *,
    fp16: bool = True,
    int8: bool = False,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 16,
    workspace_gib: int = 2,
) -> pathlib.Path:
    """Build a TensorRT engine from an ONNX model."""
    try:
        import tensorrt as trt
    except ImportError:
        log.error(
            "tensorrt package not found. Install with:\n"
            "  pip install tensorrt          # or tensorrt-cu12 / tensorrt-cu12-bindings\n\n"
            "Or use the trtexec CLI instead (no Python needed)."
        )
        sys.exit(1)

    trt_logger = trt.Logger(trt.Logger.INFO)
    log.info("TensorRT version: %s", trt.__version__)
    builder = trt.Builder(trt_logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, trt_logger)

    log.info("Parsing ONNX model: %s", onnx_path)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error("TRT ONNX parse error: %s", parser.get_error(i))
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)

    if fp16:
        if not builder.platform_has_fast_fp16:
            log.warning("Platform does not report fast FP16 — enabling anyway.")
        config.set_flag(trt.BuilderFlag.FP16)
        log.info("FP16 enabled.")
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        log.info("INT8 enabled (you may need a calibrator for best accuracy).")

    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    spatial_dims = tuple(input_tensor.shape[1:])

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        min=(min_batch, *spatial_dims),
        opt=(opt_batch, *spatial_dims),
        max=(max_batch, *spatial_dims),
    )
    config.add_optimization_profile(profile)
    log.info(
        "Optimization profile: input='%s'  min=(%d, %s)  opt=(%d, %s)  max=(%d, %s)",
        input_name,
        min_batch, ", ".join(str(d) for d in spatial_dims),
        opt_batch, ", ".join(str(d) for d in spatial_dims),
        max_batch, ", ".join(str(d) for d in spatial_dims),
    )

    log.info("Building TensorRT engine (this may take a few minutes) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        log.error("TensorRT engine build failed.")
        sys.exit(1)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    log.info("Saved TensorRT engine → %s  (%.1f MB)", engine_path, engine_path.stat().st_size / 1e6)
    return engine_path


def print_trtexec_command(
    onnx_path: pathlib.Path,
    engine_path: pathlib.Path,
    *,
    fp16: bool,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    input_name: str,
    spatial: str,
) -> None:
    """Print the equivalent trtexec CLI command for reference."""
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


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════


def _cmd_onnx(args: argparse.Namespace) -> int:
    keras_path: pathlib.Path = args.keras_model.resolve()
    config_path: pathlib.Path = args.plate_config.resolve()
    out_dir = (args.out_dir or keras_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_onnx = out_dir / f"{keras_path.stem}.onnx"

    export_to_onnx(
        keras_path,
        config_path,
        out_onnx,
        input_dtype=args.input_dtype,
        data_format=args.data_format,
        dynamic_batch=not args.no_dynamic_batch,
        simplify=not args.no_simplify,
        opset_version=args.opset,
    )
    return 0


def _cmd_trt(args: argparse.Namespace) -> int:
    onnx_path: pathlib.Path = args.onnx.resolve()
    if not onnx_path.is_file():
        log.error("ONNX file not found: %s", onnx_path)
        return 1

    engine_path = (args.engine or onnx_path.with_suffix(".engine")).resolve()

    input_name, spatial_dims = _read_onnx_input_meta(onnx_path)
    spatial_str = "x".join(str(d) for d in spatial_dims)
    log.info("ONNX input: name='%s'  shape=(N, %s)", input_name, ", ".join(str(d) for d in spatial_dims))

    print_trtexec_command(
        onnx_path,
        engine_path,
        fp16=args.trt_fp16,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        input_name=input_name,
        spatial=spatial_str,
    )

    build_trt_engine(
        onnx_path,
        engine_path,
        fp16=args.trt_fp16,
        int8=args.trt_int8,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        workspace_gib=args.workspace_gib,
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export plate-OCR models: Keras → ONNX and/or ONNX → TensorRT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── onnx subcommand ──
    onnx_p = sub.add_parser("onnx", help="Convert a Keras model to ONNX.")
    onnx_p.add_argument("--keras-model", required=True, type=pathlib.Path, help="Path to .keras model file.")
    onnx_p.add_argument("--plate-config", required=True, type=pathlib.Path, help="Path to plate_config.yaml.")
    onnx_p.add_argument("--out-dir", type=pathlib.Path, default=None, help="Output directory (default: same as --keras-model).")
    onnx_p.add_argument("--input-dtype", choices=["uint8", "float32"], default="uint8", help="ONNX input dtype (default: uint8).")
    onnx_p.add_argument(
        "--data-format",
        choices=["channels_last", "channels_first"],
        default="channels_last",
        help="Input layout: channels_last (NHWC) or channels_first (NCHW). Default: channels_last.",
    )
    onnx_p.add_argument("--no-simplify", action="store_true", help="Skip onnxslim graph simplification.")
    onnx_p.add_argument("--no-dynamic-batch", action="store_true", help="Use static batch=1 instead of dynamic.")
    onnx_p.add_argument("--opset", type=int, default=None, help="ONNX opset version (default: Keras default).")

    # ── trt subcommand ──
    trt_p = sub.add_parser("trt", help="Convert an ONNX model to a TensorRT engine.")
    trt_p.add_argument("--onnx", required=True, type=pathlib.Path, help="Path to the .onnx model.")
    trt_p.add_argument("--engine", type=pathlib.Path, default=None, help="Output .engine path (default: <onnx_stem>.engine).")
    trt_p.add_argument("--trt-fp16", action="store_true", help="Enable FP16 precision.")
    trt_p.add_argument("--trt-int8", action="store_true", help="Enable INT8 precision (needs calibrator for best results).")
    trt_p.add_argument("--min-batch", type=int, default=1, help="Min batch size for TRT profile (default: 1).")
    trt_p.add_argument("--opt-batch", type=int, default=1, help="Optimal batch size for TRT profile (default: 1).")
    trt_p.add_argument("--max-batch", type=int, default=16, help="Max batch size for TRT profile (default: 16).")
    trt_p.add_argument("--workspace-gib", type=int, default=2, help="TRT workspace memory in GiB (default: 2).")

    args = p.parse_args()

    if args.command == "onnx":
        return _cmd_onnx(args)
    if args.command == "trt":
        return _cmd_trt(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
