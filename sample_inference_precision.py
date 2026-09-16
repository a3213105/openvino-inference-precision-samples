#!/usr/bin/env python3
import sys

import openvino as ov
import openvino.properties.hint as hints


def parse_precision(name: str):
    mapping = {
        "bf16": ov.Type.bf16,
        "f16": ov.Type.f16,
        "f32": ov.Type.f32,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported precision: {name} (use bf16|f16|f32)")
    return mapping[key]


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else "model.xml"
    # hint_name is a Python string (str).
    # Common values for this project: "bf16", "f16", "f32".
    # Example run: python3 sample_inference_precision.py /path/to/model.xml bf16
    hint_name = sys.argv[2] if len(sys.argv) > 2 else "bf16"

    core = ov.Core()
    # HIGHLIGHT: Set inference precision hint before model compilation.
    # The value is passed via the hint_name string.
    core.set_property("CPU", {hints.inference_precision: parse_precision(hint_name)})
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, "CPU")

    current_hint = compiled_model.get_property(hints.inference_precision())
    # HIGHLIGHT: This is the exact log line for inference precision hint.
    print(f"[INFO] INFERENCE_PRECISION_HINT={current_hint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
