# Description:
#   TensorFlow Lite for Microcontrollers "hello world" example.
load("@rules_python//python:defs.bzl", "py_binary")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load(
    "//machina/lite/micro:build_def.bzl",
    "tflm_cc_library",
    "tflm_cc_test",
)

package(
    # Disabling layering_check because of http://b/177257332
    features = ["-layering_check"],
    licenses = ["notice"],
)

tflm_cc_library(
    name = "model",
    srcs = [
        "//machina/lite/micro/examples/hello_world/models:generated_hello_world_float_model_cc",
        "//machina/lite/micro/examples/hello_world/models:generated_hello_world_int8_model_cc",
    ],
    hdrs = [
        "//machina/lite/micro/examples/hello_world/models:generated_hello_world_float_model_hdr",
        "//machina/lite/micro/examples/hello_world/models:generated_hello_world_int8_model_hdr",
    ],
)

tflm_cc_test(
    name = "hello_world_test",
    srcs = [
        "hello_world_test.cc",
    ],
    deps = [
        ":model",
        "//machina/lite/micro:micro_framework",
        "//machina/lite/micro:micro_log",
        "//machina/lite/micro:micro_profiler",
        "//machina/lite/micro:op_resolvers",
        "//machina/lite/micro:recording_allocators",
        "//machina/lite/micro/testing:micro_test",
        "//machina/lite/schema:schema_fbs",
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    data = ["//machina/lite/micro/examples/hello_world/models:hello_world_float.tflite"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
        requirement("numpy"),
        requirement("machina"),
        "//python/tflite_micro:runtime",
    ],
)

py_binary(
    name = "evaluate_test",
    srcs = ["evaluate_test.py"],
    data = [
        "//machina/lite/micro/examples/hello_world/models:hello_world_float.tflite",
        "//machina/lite/micro/examples/hello_world/models:hello_world_int8.tflite",
    ],
    deps = [
        ":evaluate",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        requirement("numpy"),
        requirement("machina"),
    ],
)
