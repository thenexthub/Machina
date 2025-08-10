load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")

# Placeholder: load py_proto_library
load("//machina:machina.bzl", "tf_cc_binary", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_proto_library",
)
load("//machina/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//machina/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//machina/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

exports_files([
    "logging.h",
])

py_strict_binary(
    name = "visualize",
    srcs = ["visualize.py"],
    deps = [
        "//machina/lite/python:schema_py",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "visualize_lib",
    srcs = ["visualize.py"],
    deps = [
        "//machina/lite/python:schema_py",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "visualize_test",
    srcs = ["visualize_test.py"],
    deps = [
        ":test_utils",
        ":visualize_lib",
        #internal proto upb dep
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_binary(
    name = "convert_image_to_csv",
    srcs = ["convert_image_to_csv.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:io_ops",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_library(
    name = "convert_image_to_csv_lib",
    srcs = ["convert_image_to_csv.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:io_ops",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_test(
    name = "convert_image_to_csv_test",
    srcs = ["convert_image_to_csv_test.py"],
    data = ["//machina/core:image_testdata"],
    deps = [
        ":convert_image_to_csv_lib",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
    ],
)

py_strict_binary(
    name = "strip_strings",
    srcs = ["strip_strings.py"],
    deps = [
        ":flatbuffer_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_binary(
    name = "reverse_xxd_dump_from_cc",
    srcs = ["reverse_xxd_dump_from_cc.py"],
    deps = [
        ":flatbuffer_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_binary(
    name = "randomize_weights",
    srcs = ["randomize_weights.py"],
    deps = [
        ":flatbuffer_utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_library(
    name = "flatbuffer_utils",
    srcs = ["flatbuffer_utils.py"],
    deps = [
        "//machina/lite/python:schema_py",
        "//machina/lite/python:schema_util",
        "//machina/python/platform:gfile",
        "@flatbuffers//:runtime_py",
    ],
)

py_strict_test(
    name = "flatbuffer_utils_test",
    srcs = ["flatbuffer_utils_test.py"],
    deps = [
        ":flatbuffer_utils",
        ":test_utils",
        #internal proto upb dep
        "//machina/lite/python:schema_py",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    deps = [
        "//machina/lite/python:schema_py",
        "@flatbuffers//:runtime_py",
    ],
)

cc_binary(
    name = "generate_op_registrations",
    srcs = ["gen_op_registration_main.cc"],
    deps = [
        ":command_line_flags",
        ":gen_op_registration",
        "//machina/lite:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gen_op_registration",
    srcs = ["gen_op_registration.cc"],
    hdrs = ["gen_op_registration.h"],
    deps = [
        "//machina/lite:framework",
        "//machina/lite:string",
        "//machina/lite/core:framework",
        "//machina/lite/schema:schema_utils",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "gen_op_registration_test",
    srcs = ["gen_op_registration_test.cc"],
    data = [
        "//machina/lite:testdata/0_subgraphs.bin",
        "//machina/lite:testdata/2_subgraphs.bin",
        "//machina/lite:testdata/empty_model.bin",
        "//machina/lite:testdata/test_model.bin",
        "//machina/lite:testdata/test_model_broken.bin",
        "//machina/lite:testdata/test_model_versioned_ops.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":gen_op_registration",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite(
    name = "verifier",
    hdrs = ["verifier.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    tflite_deps = [
        "//machina/lite:framework_stable",
    ],
    deps = [
        ":verifier_internal",
        "//machina/lite:schema_fbs_version",
        "//machina/lite:string_util",
        "//machina/lite:util",
        "//machina/lite/core/api:error_reporter",
        "//machina/lite/core/api:op_resolver",
        "//machina/lite/core/c:common",
        "//machina/lite/core/tools:verifier",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_utils",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library_with_tflite(
    name = "verifier_internal",
    hdrs = ["verifier_internal.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    deps = ["//machina/lite/core/tools:verifier_internal"],
)

cc_library(
    name = "logging",
    hdrs = ["logging.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
)

cc_library(
    name = "tool_params",
    srcs = ["tool_params.cc"],
    hdrs = ["tool_params.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [":logging"],
)

cc_test(
    name = "tool_params_test",
    srcs = ["tool_params_test.cc"],
    copts = tflite_copts(),
    visibility = ["//visibility:private"],
    deps = [
        ":tool_params",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "command_line_flags",
    srcs = ["command_line_flags.cc"],
    hdrs = ["command_line_flags.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts(),
    deps = [
        ":logging",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "command_line_flags_test",
    srcs = ["command_line_flags_test.cc"],
    copts = tflite_copts(),
    visibility = ["//visibility:private"],
    deps = [
        ":command_line_flags",
        ":tool_params",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "list_flex_ops",
    srcs = ["list_flex_ops.cc"],
    hdrs = ["list_flex_ops.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/lite:framework",
        "//machina/lite:util",
        "//machina/lite/core:framework",
        "//machina/lite/schema:schema_utils",
        "@flatbuffers",
        "@jsoncpp_git//:jsoncpp",
    ],
)

# This tool list flex ops and kernels inside a TFLite file.
# It is used to generate header file for selective registration.
tf_cc_binary(
    name = "list_flex_ops_main",
    srcs = ["list_flex_ops_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":command_line_flags",
        ":list_flex_ops",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "list_flex_ops_main_lib",
    srcs = ["list_flex_ops_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":command_line_flags",
        ":list_flex_ops",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "list_flex_ops_no_kernel",
    srcs = ["list_flex_ops_no_kernel.cc"],
    hdrs = ["list_flex_ops.h"],
    deps = [
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/schema:schema_utils",
        "@jsoncpp_git//:jsoncpp",
    ],
)

tf_cc_binary(
    name = "list_flex_ops_no_kernel_main",
    srcs = ["list_flex_ops_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":command_line_flags",
        ":list_flex_ops_no_kernel",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "list_flex_ops_test",
    srcs = ["list_flex_ops_test.cc"],
    data = [
        "//machina/lite:testdata/0_subgraphs.bin",
        "//machina/lite:testdata/multi_add_flex.bin",
        "//machina/lite:testdata/softplus_flex.bin",
        "//machina/lite:testdata/test_model.bin",
        "//machina/lite:testdata/test_model_broken.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":list_flex_ops",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:protobuf",
        "//machina/core/platform:resource_loader",
        "//machina/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

py_strict_binary(
    name = "zip_files",
    srcs = ["zip_files.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        ":logging",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    copts = tflite_copts(),
    deps = [
        ":utils",
        "//machina/lite/c:common",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "model_loader",
    srcs = ["model_loader.cc"],
    hdrs = ["model_loader.h"],
    deps = [
        "//machina/lite:minimal_logging",
        "//machina/lite/core:model_builder",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "model_loader_test",
    srcs = ["model_loader_test.cc"],
    data = ["@tflite_mobilenet_float//:mobilenet_v1_1.0_224.tflite"],
    deps = [
        ":model_loader",
        "//machina/lite:model_builder",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers//:runtime_cc",
    ],
)

tflite_portable_test_suite()

# copybara:uncomment_begin(google-only)
# tf_proto_library(
#     name = "op_kernel_set_proto",
#     srcs = ["op_kernel_set.proto"],
# )
#
# py_proto_library(
#     name = "op_kernel_set_py_pb2",
#     deps = [":op_kernel_set_proto"],
# )
# copybara:uncomment_end
