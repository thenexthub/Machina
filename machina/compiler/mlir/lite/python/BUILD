load("@local_xla//xla/tsl/platform:build_config_root.bzl", "if_pywrap")
load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "tf_python_pybind_extension")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

licenses(["notice"])

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        ":friends",
        "//machina:__pkg__",
    ],
)

package_group(
    name = "friends",
    packages = [
        "//learning/brain/mobile/lite/tooling/model_analyzer/...",
        "//learning/brain/mobile/lite/tools/analyzer/...",
        "//machina/lite/python/...",
        "//machina/lite/python/converter/...",
        "//machina/lite/toco/...",
    ],
)

cc_library(
    name = "tf_tfl_flatbuffer_helpers",
    srcs = ["tf_tfl_flatbuffer_helpers.cc"],
    hdrs = ["tf_tfl_flatbuffer_helpers.h"],
    deps = [
        "//machina/compiler/mlir/lite:common",
        "//machina/compiler/mlir/lite:converter_flags_proto_cc",
        "//machina/compiler/mlir/lite:model_flags_proto_cc",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//machina/compiler/mlir/lite:types_proto_cc",
        "//machina/compiler/mlir/lite/quantization/common/quantization_lib:quantization_config",
        "//machina/compiler/mlir/lite/tools/optimize:reduced_precision_metadata",
        "//machina/compiler/mlir/quantization/machina/python:py_function_lib",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "graphdef_to_tfl_flatbuffer",
    srcs = ["graphdef_to_tfl_flatbuffer.cc"],
    hdrs = [
        "graphdef_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":tf_tfl_flatbuffer_helpers",
        "//machina/compiler/mlir/lite:common",
        "//machina/compiler/mlir/lite:converter_flags_proto_cc",
        "//machina/compiler/mlir/lite:model_flags_proto_cc",
        "//machina/compiler/mlir/lite:types_proto_cc",
        "//machina/compiler/mlir/lite/quantization/common/quantization_lib:quantization_config",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/machina/translate/tools:parsers",
        "//machina/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "saved_model_to_tfl_flatbuffer",
    srcs = ["saved_model_to_tfl_flatbuffer.cc"],
    hdrs = [
        "saved_model_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":tf_tfl_flatbuffer_helpers",
        "//machina/compiler/mlir/lite:common",
        "//machina/compiler/mlir/lite:converter_flags_proto_cc",
        "//machina/compiler/mlir/lite:model_flags_proto_cc",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite:tf_to_tfl_flatbuffer",
        "//machina/compiler/mlir/lite:types_proto_cc",
        "//machina/compiler/mlir/lite/quantization/common/quantization_lib:quantization_config",
        "//machina/compiler/mlir/quantization/machina/python:py_function_lib",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "jax_to_tfl_flatbuffer",
    srcs = ["jax_to_tfl_flatbuffer.cc"],
    hdrs = [
        "jax_to_tfl_flatbuffer.h",
    ],
    deps = [
        ":tf_tfl_flatbuffer_helpers",
        "//machina/compiler/mlir/lite:common",
        "//machina/compiler/mlir/lite:converter_flags_proto_cc",
        "//machina/compiler/mlir/lite:model_flags_proto_cc",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite:types_proto_cc",
        "//machina/compiler/mlir/lite/quantization/common/quantization_lib:quantization_config",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/utility",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/hlo/parser:hlo_parser",
        "@local_xla//xla/hlo/translate:stablehlo",
        "@local_xla//xla/service:hlo_proto_cc",
    ],
)

# Smaller version of flatbuffer_translate which only converts flatbuffer to MLIR.
cc_library(
    name = "flatbuffer_to_mlir",
    srcs = [
        "flatbuffer_to_mlir.cc",
    ],
    hdrs = [
        "flatbuffer_to_mlir.h",
    ],
    deps = [
        "//machina/compiler/mlir/lite:flatbuffer_import",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TranslateLib",
    ],
)

py_strict_library(
    name = "wrap_converter",
    srcs = [
        "wrap_converter.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_converter_api",
        "//machina/compiler/mlir/quantization/machina/python:py_function_lib_py",
        "//machina/python:pywrap_machina",
    ],
)

config_setting(
    name = "tflite_convert_with_select_tf_ops",
    define_values = {"tflite_convert_with_select_tf_ops": "true"},
    visibility = ["//visibility:private"],
)

filegroup(
    name = "converter_python_api_hdrs",
    srcs = [
        "converter_python_api.h",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "converter_python_api",
    srcs = ["converter_python_api.cc"],
    hdrs = ["converter_python_api.h"],
    features = ["-parse_headers"],
    visibility = [
        "//machina/python:__subpackages__",
    ],
    deps = [
        ":flatbuffer_to_mlir",
        ":graphdef_to_tfl_flatbuffer",
        ":jax_to_tfl_flatbuffer",
        ":saved_model_to_tfl_flatbuffer",
        "//machina/c:kernels",
        "//machina/c:tf_status_headers",
        "//machina/compiler/mlir/lite:converter_flags_proto_cc",
        "//machina/compiler/mlir/lite:model_flags_proto_cc",
        "//machina/compiler/mlir/lite:types_proto_cc",
        "//machina/compiler/mlir/lite/core:absl_error_model_builder",
        "//machina/compiler/mlir/lite/debug:debug_options_proto_cc",
        "//machina/compiler/mlir/lite/metrics:error_collector",
        "//machina/compiler/mlir/lite/python/interpreter_wrapper:python_error_reporter",
        "//machina/compiler/mlir/lite/python/interpreter_wrapper:python_utils",
        "//machina/compiler/mlir/lite/quantization/lite:quantize_model",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/lite/sparsity:sparsify_model",
        "//machina/compiler/mlir/quantization/machina/python:py_function_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//:protobuf_headers",
        "@flatbuffers//:runtime_cc",
        "@local_xla//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
        "@local_xla//xla/tsl/platform:status",
    ] + select({
        # This is required when running `tflite_convert` from `bazel`.
        # It requires to link with TensorFlow Ops to get the op definitions.
        ":tflite_convert_with_select_tf_ops": [
            "//machina/core:ops",
        ],
        "//conditions:default": [],
    }),
    alwayslink = True,
)

tf_python_pybind_extension(
    name = "_pywrap_converter_api",
    srcs = [
        "converter_python_api_wrapper.cc",
    ],
    hdrs = [":converter_python_api_hdrs"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_converter_api.pyi",
    ],
    visibility = [
        "//machina/python:__pkg__",
    ],
    deps = [
        "//machina/compiler/mlir/quantization/machina/python:py_function_lib",
        "//machina/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ] + if_pywrap([":converter_python_api"]),
)
