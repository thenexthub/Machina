load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

config_setting(
    name = "tflite_convert_with_select_tf_ops",
    define_values = {"tflite_convert_with_select_tf_ops": "true"},
    visibility = ["//visibility:private"],
)

filegroup(
    name = "toco_python_api_hdrs",
    srcs = [
        "toco_python_api.h",
    ],
    visibility = [
        "//machina/python:__subpackages__",
    ],
)

cc_library(
    name = "toco_python_api",
    srcs = ["toco_python_api.cc"],
    hdrs = ["toco_python_api.h"],
    features = ["-parse_headers"],
    visibility = [
        "//machina/python:__subpackages__",
    ],
    deps = [
        "//machina/c:kernels",
        "//machina/c:tf_status_headers",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/lite:model_builder",
        "//machina/lite/core/api",
        "//machina/lite/core/c:common",
        "//machina/lite/python/interpreter_wrapper:python_error_reporter",
        "//machina/lite/python/interpreter_wrapper:python_utils",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/toco:model",
        "//machina/lite/toco:model_flags_proto_cc",
        "//machina/lite/toco:toco_convert",
        "//machina/lite/toco:toco_flags_proto_cc",
        "//machina/lite/toco:toco_graphviz_dump_options",
        "//machina/lite/toco:toco_port",
        "//machina/lite/toco:toco_tooling",
        "//machina/lite/toco:tooling_util",
        "//machina/lite/toco:types_proto_cc",
        "//machina/lite/toco/logging:conversion_log_util",
        "//machina/lite/toco/logging:toco_conversion_log_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf_headers",
        "@flatbuffers//:runtime_cc",
        "@local_tsl//tsl/platform:status",
        "@local_xla//third_party/python_runtime:headers",  # build_cleaner: keep; DNR: b/35864863
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

# Compatibility stub. Remove when internal customers moved.
py_strict_library(
    name = "machina_wrap_toco",
    srcs = ["machina_wrap_toco.py"],
    visibility = ["//learning/expander/pod/deep_pod/utils:__subpackages__"],
    deps = [
        "//machina/python:_pywrap_toco_api",
    ],
)
