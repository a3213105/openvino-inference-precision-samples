# OpenVINO Inference Precision Hint Samples

This folder contains three minimal samples that print only `INFERENCE_PRECISION_HINT`.

## Files

- `sample_inference_precision.c`
- `sample_inference_precision.cpp`
- `sample_inference_precision.py`

## What To Copy Into Client Code

Each sample has `HIGHLIGHT` comments for both steps:

- setting inference precision before compilation
- printing `INFERENCE_PRECISION_HINT` after compilation

### C

`infer_precision_hint` is a C string (`const char*`).

Common inputs for this project:

- `"bf16"`
- `"f16"`
- `"f32"`

1. Set inference precision hint (before compilation):

```c
ov_core_set_property(core,
                     "CPU",
                     ov_property_key_hint_inference_precision,
                     infer_precision_hint);
```

2. Query property:

```c
char* current_hint = NULL;
ov_compiled_model_get_property(compiled_model,
                               ov_property_key_hint_inference_precision,
                               &current_hint);
```

3. Print and free:

```c
// HIGHLIGHT: This is the exact log line for inference precision hint.
printf("[INFO] INFERENCE_PRECISION_HINT=%s\n", current_hint);
ov_free(current_hint);
```

### C++

`hint` is a C++ string (`std::string`).

Common inputs for this project:

- `"bf16"`
- `"f16"`
- `"f32"`

1. Set inference precision hint (before compilation):

```cpp
core.set_property("CPU", ov::hint::inference_precision(parse_precision(hint)));
```

2. Query and print:

```cpp
ov::element::Type current_hint = compiled_model.get_property(ov::hint::inference_precision);
// HIGHLIGHT: This is the exact log line for inference precision hint.
std::cout << "[INFO] INFERENCE_PRECISION_HINT=" << current_hint.get_type_name() << std::endl;
```

### Python

`hint_name` is a Python string (`str`).

Common inputs for this project:

- `"bf16"`
- `"f16"`
- `"f32"`

1. Set inference precision hint (before compilation):

```python
core.set_property("CPU", {hints.inference_precision: parse_precision(hint_name)})
```

2. Query and print:

```python
current_hint = compiled_model.get_property(hints.inference_precision())
# HIGHLIGHT: This is the exact log line for inference precision hint.
print(f"[INFO] INFERENCE_PRECISION_HINT={current_hint}")
```

## Suggested Insertion Point

Place the setting lines immediately before `compile_model(...)`.

Place the query/print lines immediately after `compile_model(...)` succeeds. This guarantees the compiled model exists and the property can be queried safely.

## Confirm With benchmark_app

You can also verify inference precision with OpenVINO `benchmark_app` directly.

### Method

1. Set the target precision using `-infer_precision`.
2. Set `-shape` based on the real input names and dimensions of your model.
3. For multi-input models, pass all input shapes in one `-shape` string, separated by commas.
4. Check runtime parameters section for `INFERENCE_PRECISION_HINT`.

### infer_precision Supported Input List

`benchmark_app -h` documents the `-infer_precision` option and shows two accepted input formats:

1. Single value (all target devices use the same precision):

```bash
-infer_precision bf16
```

2. Per-device mapping (set precision per device):

```bash
-infer_precision CPU:bf16,GPU:f32
```

For this project, the commonly used and validated values are:

- `bf16`
- `f16`
- `f32`

Note:

- Actual accepted values depend on device plugin and OpenVINO version.
- If a value is unsupported on a device, `benchmark_app` may fail or fallback; always confirm by checking the printed `INFERENCE_PRECISION_HINT` line.

Important:

- The `-shape` value in examples is only a template. You must replace it for each model.
- Input names in `-shape` must match model input tensor names exactly.
- Recommended: run once and check the `Model inputs` section printed by `benchmark_app`, then fill exact names/shapes.

### Example

```bash
benchmark_app \
  -m /path/to/model.xml \
  -d CPU \
  -shape "input_name[1,3,224,224]" \
  -infer_precision bf16 \
  -hint latency \
  -api sync \
  -t 1
```

### Multi-Input Shape Example

```bash
benchmark_app \
  -m /path/to/model.xml \
  -d CPU \
  -shape "input_ids[1,128],attention_mask[1,128],token_type_ids[1,128]" \
  -infer_precision f16 \
  -hint latency \
  -api sync \
  -t 1
```

Another multi-input example (vision + metadata):

```bash
benchmark_app \
  -m /path/to/model.xml \
  -d CPU \
  -shape "image[1,3,224,224],aux_feature[1,32]" \
  -infer_precision bf16 \
  -hint latency \
  -api sync \
  -t 1
```

Expected key line in output:

```text
[ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'bfloat16'>
```

### Fast Check In Script/CI

```bash
benchmark_app ... -infer_precision f16 ... 2>&1 | grep -E "INFERENCE_PRECISION_HINT"
```

If the printed value matches your expected type (`bfloat16`, `float16`, `float32`), the runtime hint is configured as intended.

## Quick Run

Python:

```bash
python3 sample_inference_precision.py /path/to/model.xml bf16
```

C++ (example, adjust include/lib paths for your OpenVINO installation):

```bash
g++ -std=c++17 sample_inference_precision.cpp -o sample_cpp \
  $(pkg-config --cflags --libs openvino)
./sample_cpp /path/to/model.xml bf16
```

C (example, adjust include/lib paths for your OpenVINO installation):

```bash
gcc sample_inference_precision.c -o sample_c \
  $(pkg-config --cflags --libs openvino-c)
./sample_c /path/to/model.xml bf16
```
