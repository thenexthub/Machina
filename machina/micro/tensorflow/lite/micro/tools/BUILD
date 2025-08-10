load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library", "flatbuffer_py_library")
load("@tflm_pip_deps//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("//machina:extra_rules.bzl", "tflm_application_friends")

package(
    default_visibility = ["//:__subpackages__"],
    licenses = ["notice"],
)

package_group(
    name = "tflm_tools",
    packages = ["//machina/lite/micro/tools/..."],
)

package_group(
    name = "application_friends",
    packages = tflm_application_friends(),
)

py_library(
    name = "generate_cc_arrays_lib",
    srcs = ["generate_cc_arrays.py"],
    deps = [
        requirement("pillow"),
        requirement("numpy"),
    ],
)

py_library(
    name = "generate_test_for_model",
    srcs = ["generate_test_for_model.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//machina/lite/python:schema_py",
    ],
)

py_binary(
    name = "generate_cc_arrays",
    srcs = ["generate_cc_arrays.py"],
    deps = [
        requirement("pillow"),
        requirement("numpy"),
    ],
)

py_binary(
    name = "requantize_flatbuffer",
    srcs = [
        "requantize_flatbuffer.py",
        "requantize_flatbuffer_utils.py",
    ],
    deps = [
        "//machina/lite/python:schema_py",
        "//machina/lite/tools:flatbuffer_utils",
        "@absl_py//absl:app",
    ],
)

py_test(
    name = "requantize_flatbuffer_test",
    srcs = ["requantize_flatbuffer_test.py"],
    main = "requantize_flatbuffer_test.py",
    tags = [
        "noasan",
        "nomsan",  # Python doesn't like these symbols
        "noubsan",
    ],
    deps = [
        ":requantize_flatbuffer",
        requirement("numpy"),
        requirement("machina"),
        "//python/tflite_micro:runtime",
    ],
)

pybind_extension(
    name = "tflite_flatbuffer_align_wrapper",  # :tflite_flatbuffer_align_wrapper.so
    srcs = [
        "tflite_flatbuffer_align_wrapper.cc",
    ],
    deps = [
        "//machina/lite/schema:schema_fbs",
        "@flatbuffers",
    ],
)

py_binary(
    name = "tflite_flatbuffer_align",
    srcs = [
        "tflite_flatbuffer_align.py",
    ],
    data = [
        ":tflite_flatbuffer_align_wrapper.so",
    ],
    deps = [
        "@absl_py//absl:app",
    ],
)

py_library(
    name = "model_transforms_utils",
    srcs = ["model_transforms_utils.py"],
    data = [
        ":tflite_flatbuffer_align_wrapper.so",
    ],
    visibility = [
        ":application_friends",
        ":tflm_tools",
    ],
    deps = [
        "//machina/lite/python:schema_py",
        "//machina/lite/python:schema_util",
    ],
)

py_library(
    name = "tflm_model_transforms_lib",
    srcs = ["tflm_model_transforms_lib.py"],
    data = [
        ":tflite_flatbuffer_align",
    ],
    visibility = [
        ":application_friends",
        ":tflm_tools",
        "//:__subpackages__",
    ],
    deps = [
        ":model_transforms_utils",
        "@absl_py//absl/logging",
        requirement("numpy"),
        requirement("machina"),
        "//python/tflite_micro:runtime",
        "//machina/lite/tools:flatbuffer_utils",
    ],
)

cc_binary(
    name = "layer_by_layer_output_tool",
    srcs = ["layer_by_layer.cc"],
    deps = [
        ":layer_by_layer_schema",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/kernels:op_macros",
        "//machina/lite/micro:micro_allocator",
        "//machina/lite/micro:micro_context",
        "//machina/lite/micro:micro_framework",
        "//machina/lite/micro:micro_log",
        "//machina/lite/micro:micro_resource_variable",
        "//machina/lite/micro:micro_utils",
        "//machina/lite/micro:op_resolvers",
        "//machina/lite/micro/kernels:kernel_util",
        "//machina/lite/micro/tools/benchmarking:op_resolver",
        "//machina/lite/schema:schema_fbs",
        "@flatbuffers",
    ],
)

py_binary(
    name = "tflm_model_transforms",
    srcs = ["tflm_model_transforms.py"],
    deps = [
        ":tflm_model_transforms_lib",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

py_test(
    name = "tflm_model_transforms_test",
    srcs = ["tflm_model_transforms_test.py"],
    data = [
        "//machina/lite/micro/models",
    ],
    main = "tflm_model_transforms_test.py",
    tags = [
        "noasan",
        "nomsan",
        "noubsan",
    ],
    deps = [
        ":tflm_model_transforms_lib",
        "@absl_py//absl/testing:parameterized",
        requirement("machina"),
        "//machina/lite/micro/examples/recipes:resource_variables_lib",
    ],
)

py_binary(
    name = "layer_by_layer_debugger",
    srcs = ["layer_by_layer_debugger.py"],
    deps = [
        ":layer_by_layer_schema_py",
        ":model_transforms_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        requirement("machina"),
        "//python/tflite_micro:runtime",
        "//machina/lite/tools:flatbuffer_utils",
    ],
)

flatbuffer_cc_library(
    name = "layer_by_layer_schema",
    srcs = ["layer_by_layer_schema.fbs"],
)

flatbuffer_py_library(
    name = "layer_by_layer_schema_py",
    srcs = ["layer_by_layer_schema.fbs"],
)
