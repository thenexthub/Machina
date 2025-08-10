# Description:
#   TensorFlow Lite microcontroller benchmarks.

load(
    "//machina/lite/micro:build_def.bzl",
    "tflm_cc_binary",
    "tflm_cc_library",
)

package(
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

package_group(
    name = "micro_top_level",
    packages = ["//machina/lite/micro"],
)

tflm_cc_library(
    name = "micro_benchmark",
    hdrs = [
        "micro_benchmark.h",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//machina/lite/micro:micro_framework",
        "//machina/lite/micro:micro_log",
        "//machina/lite/micro:micro_resource_variable",
        "//machina/lite/micro:micro_time",
        "//machina/lite/micro:op_resolvers",
        "//machina/lite/micro:recording_allocators",
    ],
)

tflm_cc_library(
    name = "keyword_scrambled_model_data",
    srcs = [
        "//machina/lite/micro/models:generated_keyword_scrambled_model_cc",
    ],
    hdrs = [
        "//machina/lite/micro/models:generated_keyword_scrambled_model_hdr",
    ],
    visibility = [
        ":micro_top_level",
    ],
)

tflm_cc_binary(
    name = "keyword_benchmark",
    srcs = ["keyword_benchmark.cc"],
    deps = [
        ":keyword_scrambled_model_data",
        ":micro_benchmark",
        "//machina/lite/c:common",
        "//machina/lite/micro:micro_framework",
        "//machina/lite/micro:micro_log",
        "//machina/lite/micro:micro_profiler",
        "//machina/lite/micro:op_resolvers",
        "//machina/lite/micro:system_setup",
    ],
)

tflm_cc_library(
    name = "keyword_scrambled_8bit_model_data",
    srcs = [
        "//machina/lite/micro/models:generated_keyword_scrambled_8bit_model_cc",
    ],
    hdrs = [
        "//machina/lite/micro/models:generated_keyword_scrambled_8bit_model_hdr",
    ],
    visibility = ["//visibility:private"],
)

tflm_cc_binary(
    name = "keyword_benchmark_8bit",
    srcs = ["keyword_benchmark_8bit.cc"],
    deps = [
        ":keyword_scrambled_8bit_model_data",
        ":micro_benchmark",
        "//machina/lite/c:common",
        "//machina/lite/micro:micro_framework",
        "//machina/lite/micro:micro_log",
        "//machina/lite/micro:op_resolvers",
        "//machina/lite/micro:system_setup",
    ],
)

tflm_cc_binary(
    name = "person_detection_benchmark",
    srcs = ["person_detection_benchmark.cc"],
    deps = [
        ":micro_benchmark",
        "//machina/lite/c:common",
        "//machina/lite/micro:micro_framework",
        "//machina/lite/micro:micro_log",
        "//machina/lite/micro:micro_utils",
        "//machina/lite/micro:op_resolvers",
        "//machina/lite/micro:system_setup",
        "//machina/lite/micro/examples/person_detection:model_settings",
        "//machina/lite/micro/examples/person_detection:person_detect_model_data",
        "//machina/lite/micro/examples/person_detection:simple_images_test_data",
        "//machina/lite/schema:schema_fbs",
    ],
)
