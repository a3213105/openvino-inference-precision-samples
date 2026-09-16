#include <stdio.h>

#include <openvino/c/openvino.h>

int main(int argc, char** argv) {
    const char* model_path = (argc > 1) ? argv[1] : "model.xml";
    // infer_precision_hint is a C string (const char*).
    // Common values for this project: "bf16", "f16", "f32".
    // Example run: ./sample_c /path/to/model.xml bf16
    const char* infer_precision_hint = (argc > 2) ? argv[2] : "bf16";

    ov_core_t* core = NULL;
    ov_compiled_model_t* compiled_model = NULL;
    char* current_hint = NULL;

    if (ov_core_create(&core) != OK) {
        fprintf(stderr, "[ERROR] ov_core_create failed\n");
        return 1;
    }

    // HIGHLIGHT: Set inference precision hint before model compilation.
    // The value is passed as a string via infer_precision_hint.
    if (ov_core_set_property(core,
                             "CPU",
                             ov_property_key_hint_inference_precision,
                             infer_precision_hint) != OK) {
        fprintf(stderr, "[ERROR] set INFERENCE_PRECISION_HINT failed: %s\n", infer_precision_hint);
        ov_core_free(core);
        return 1;
    }

    if (ov_core_compile_model_from_file(core,
                                        model_path,
                                        "CPU",
                                        0,
                                        &compiled_model) != OK) {
        fprintf(stderr, "[ERROR] compile model failed: %s\n", model_path);
        ov_core_free(core);
        return 1;
    }

    if (ov_compiled_model_get_property(compiled_model,
                                       ov_property_key_hint_inference_precision,
                                       &current_hint) == OK &&
        current_hint != NULL) {
        // HIGHLIGHT: This is the exact log line for inference precision hint.
        printf("[INFO] INFERENCE_PRECISION_HINT=%s\n", current_hint);
        ov_free(current_hint);
    } else {
        fprintf(stderr, "[WARN] failed to read INFERENCE_PRECISION_HINT\n");
    }

    ov_compiled_model_free(compiled_model);
    ov_core_free(core);
    return 0;
}
