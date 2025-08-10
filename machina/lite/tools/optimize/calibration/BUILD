load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina/lite:build_def.bzl", "tflite_copts")
load("//machina/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "builtin_logging_op",
    srcs = ["builtin_logging_ops/lstm.cc"],
    hdrs = ["builtin_logging_ops/lstm.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_logger",
        "//machina/lite:framework",
        "//machina/lite/core:framework_stable",
        "//machina/lite/core/api",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels:lstm_shared",
        "//machina/lite/kernels:op_macros",
        "//machina/lite/kernels/internal:reference",
        "//machina/lite/kernels/internal:tensor_utils",
        "@ruy//ruy/profiler:instrumentation",
    ],
)

cc_library(
    name = "custom_logging_op",
    srcs = ["custom_logging_ops/lstm.cc"],
    hdrs = ["custom_logging_ops/lstm.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_logger",
        "//machina/lite:framework",
        "//machina/lite/core:framework_stable",
        "//machina/lite/core/api",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels:lstm_shared",
        "//machina/lite/kernels:op_macros",
        "//machina/lite/kernels/internal:reference",
        "//machina/lite/kernels/internal:tensor_utils",
    ],
)

cc_library(
    name = "calibrator_lib",
    srcs = ["calibrator.cc"],
    hdrs = ["calibrator.h"],
    copts = tflite_copts(),
    deps = [
        ":builtin_logging_op",
        ":calibration_common",
        ":calibration_logger",
        ":calibration_reader",
        ":custom_logging_op",
        ":logging_op",
        ":logging_op_resolver",
        "//machina/compiler/mlir/lite:allocation",
        "//machina/compiler/mlir/lite/schema:schema_utils",
        "//machina/lite:framework",
        "//machina/lite:minimal_logging",
        "//machina/lite:string",
        "//machina/lite/core:framework",
        "//machina/lite/core/api",
        "//machina/lite/core/c:common",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "calibrator_test",
    srcs = ["calibrator_test.cc"],
    args = [
        "--test_model_file=$(location //machina/lite:testdata/multi_add.bin)",
    ],
    data = [
        "//machina/lite:testdata/call_once_mul.bin",
        "//machina/lite:testdata/custom_lstm.bin",
        "//machina/lite:testdata/lstm.bin",
        "//machina/lite:testdata/multi_add.bin",
        "//machina/lite:testdata/multi_subgraphs_while.bin",
        "//machina/lite:testdata/unidirectional_sequence_lstm.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":calibration_reader",
        ":calibrator_lib",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/lite:framework",
        "//machina/lite:string",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core:framework",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "logging_op_resolver",
    srcs = ["logging_op_resolver.cc"],
    hdrs = ["logging_op_resolver.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_common",
        "//machina/lite:framework",
        "//machina/lite:util",
        "//machina/lite/c:common",
        "//machina/lite/core/api",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "logging_op_resolver_test",
    srcs = ["logging_op_resolver_test.cc"],
    deps = [
        ":calibration_common",
        ":logging_op_resolver",
        "//machina/lite:framework",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "calibration_reader",
    srcs = ["calibration_reader.cc"],
    hdrs = ["calibration_reader.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_logger",
        "//machina/lite:framework",
        "//machina/lite/c:common",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "calibration_logger",
    srcs = ["calibration_logger.cc"],
    hdrs = ["calibration_logger.h"],
    copts = tflite_copts(),
    deps = [
        "//machina/lite:framework",
        "//machina/lite:minimal_logging",
        "//machina/lite/c:c_api_types",
        "//machina/lite/core/api",
        "//machina/lite/core/c:common",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "calibration_common",
    hdrs = ["calibration_common.h"],
    copts = tflite_copts(),
    deps = [
        "//machina/lite:framework",
    ],
)

cc_library(
    name = "logging_op",
    hdrs = ["logging_op.h"],
    copts = tflite_copts(),
    deps = [
        ":calibration_logger",
        "//machina/lite/core/c:common",
    ],
)

tflite_portable_test_suite()
