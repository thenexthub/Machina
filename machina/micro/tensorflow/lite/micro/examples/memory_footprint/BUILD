load(
    "//machina/lite/micro:build_def.bzl",
    "generate_cc_arrays",
    "tflm_cc_library",
)

package(
    licenses = ["notice"],
)

generate_cc_arrays(
    name = "generated_simple_add_model_cc",
    src = "models/simple_add_model.tflite",
    out = "models/simple_add_model_model_data.cc",
)

generate_cc_arrays(
    name = "generated_simple_add_model_hdr",
    src = "models/simple_add_model.tflite",
    out = "models/simple_add_model_model_data.h",
)

tflm_cc_library(
    name = "simple_add_model_data",
    srcs = [
        ":generated_simple_add_model_cc",
    ],
    hdrs = [
        ":generated_simple_add_model_hdr",
    ],
)

cc_binary(
    name = "baseline_memory_footprint",
    srcs = ["baseline_memory_footprint.cc"],
    deps = [
        "//machina/lite/c:common",
        "//machina/lite/micro:system_setup",
    ],
)

cc_binary(
    name = "interpreter_memory_footprint",
    srcs = ["interpreter_memory_footprint.cc"],
    deps = [
        ":simple_add_model_data",
        "//machina/lite/c:common",
        "//machina/lite/micro:micro_profiler",
        "//machina/lite/micro:op_resolvers",
        "//machina/lite/micro:system_setup",
        "//machina/lite/micro/benchmarks:micro_benchmark",
    ],
)
