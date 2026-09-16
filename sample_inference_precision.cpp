#include <iostream>
#include <stdexcept>
#include <string>

#include <openvino/openvino.hpp>

static void validate_precision(const std::string& s) {
    if (s == "bf16" || s == "f16" || s == "f32") {
        return;
    }
    throw std::runtime_error("Unsupported precision: " + s + " (use bf16|f16|f32)");
}

int main(int argc, char** argv) {
    try {
        const std::string model_path = (argc > 1) ? argv[1] : "model.xml";
        // hint is a C++ string (std::string).
        // Common values for this project: "bf16", "f16", "f32".
        // Example run: ./sample_cpp /path/to/model.xml bf16
        const std::string hint = (argc > 2) ? argv[2] : "bf16";
        validate_precision(hint);

        ov::Core core;
        // HIGHLIGHT: Set inference precision hint before model compilation.
        // The value is passed via the hint string.
        core.set_property("CPU", ov::AnyMap{{ov::hint::inference_precision.name(), hint}});
        auto model = core.read_model(model_path);
        auto compiled_model = core.compile_model(model, "CPU");

        ov::element::Type current_hint = compiled_model.get_property(ov::hint::inference_precision);
        // HIGHLIGHT: This is the exact log line for inference precision hint.
        std::cout << "[INFO] INFERENCE_PRECISION_HINT=" << current_hint.get_type_name() << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << std::endl;
        return 1;
    }
}
