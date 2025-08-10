load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "reduced_precision_support",
    srcs = [],
    hdrs = [
        "reduced_precision_support.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/lite/tools/optimize:reduced_precision_metadata",
    ],
)

tf_cc_test(
    name = "reduced_precision_support_test",
    srcs = ["reduced_precision_support_test.cc"],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":reduced_precision_support",
        "//machina/core/platform:platform_port",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_utils",
        "//machina/lite/testing:util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

cc_library(
    name = "modify_model_interface",
    srcs = ["modify_model_interface.cc"],
    hdrs = ["modify_model_interface.h"],
    deps = [
        ":model_utils",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "modify_model_interface_test",
    srcs = ["modify_model_interface_test.cc"],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":modify_model_interface",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_binary(
    name = "modify_model_interface_main",
    srcs = ["modify_model_interface_main.cc"],
    deps = [
        ":modify_model_interface",
        ":quantize_model",
    ],
)

cc_library(
    name = "quantization_wrapper_utils",
    srcs = ["quantization_wrapper_utils.cc"],
    hdrs = ["quantization_wrapper_utils.h"],
    deps = [
        ":operator_property",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/api",
        "//machina/lite/schema:schema_fbs",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "quantization_wrapper_utils_test",
    srcs = ["quantization_wrapper_utils_test.cc"],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantization_wrapper_utils",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "quantization_wrapper_utils_custom_test",
    srcs = [
        "quantization_wrapper_utils.cc",
        "quantization_wrapper_utils.h",
        "quantization_wrapper_utils_custom_test.cc",
    ],
    defines = [
        "TFLITE_CUSTOM_LSTM",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":operator_property",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/api",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantization_wrapper",
    srcs = ["quantization_wrapper.cc"],
    hdrs = ["quantization_wrapper.h"],
    deps = [
        ":quantization_wrapper_utils",
        "//machina/lite:framework",
        "//machina/lite/core/api",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/tools/optimize:quantize_model",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantization_utils",
    srcs = ["quantization_utils.cc"],
    hdrs = ["quantization_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":model_utils",
        "//machina/lite:framework",
        "//machina/lite:minimal_logging",
        "//machina/lite/c:c_api_types",
        "//machina/lite/core/api",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels/internal:cppmath",
        "//machina/lite/kernels/internal:quantization_util",
        "//machina/lite/kernels/internal:tensor_utils",
        "//machina/lite/kernels/internal:types",
        "//machina/lite/schema:schema_fbs",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "model_utils",
    srcs = ["model_utils.cc"],
    hdrs = ["model_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":operator_property",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/kernels/internal:tensor_utils",
        "//machina/lite/kernels/internal:types",
        "//machina/lite/schema:schema_conversion_utils",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "model_utils_test",
    srcs = ["model_utils_test.cc"],
    tags = [
        "tflite_not_portable_ios",
    ],
    deps = [
        ":model_utils",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

alias(
    name = "operator_property",
    actual = "//machina/compiler/mlir/lite/tools/optimize:operator_property",
)

tf_cc_test(
    name = "quantization_utils_test",
    srcs = ["quantization_utils_test.cc"],
    args = [
        "--test_model_file=$(location //machina/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        "//machina/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantization_utils",
        "//machina/compiler/mlir/lite/quantization/lite:test_util",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_utils",
        "//machina/lite/testing:util",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@flatbuffers",
    ],
)

cc_library(
    name = "quantize_model",
    srcs = ["quantize_model.cc"],
    hdrs = ["quantize_model.h"],
    deps = [
        ":model_utils",
        ":operator_property",
        ":quantization_utils",
        "//machina/lite:framework",
        "//machina/lite:util",
        "//machina/lite/core:framework",
        "//machina/lite/core/api",
        "//machina/lite/kernels/internal:cppmath",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_utils",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "quantize_model_test",
    srcs = ["quantize_model_test.cc"],
    args = [
        "--test_model_file=$(location //machina/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        "//machina/compiler/mlir/lite/quantization/lite:testdata/add_with_const_input.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/argmax.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/broadcast_to.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/concat.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/fc.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/fc_qat.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/gather_nd.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/lstm_calibrated.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/lstm_calibrated2.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/lstm_quantized.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/lstm_quantized2.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/maximum.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/minimum.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/mixed.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/mixed16x8.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/multi_input_add_reshape.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/pack.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/resource_vars_calibrated.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/single_avg_pool_min_minus_5_max_plus_5.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/single_conv_no_bias.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_minus_127_max_plus_127.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/single_softmax_min_minus_5_max_plus_5.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/split.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/svdf_calibrated.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/svdf_quantized.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/transpose.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/unidirectional_sequence_lstm_calibrated.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/unidirectional_sequence_lstm_quantized.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/unpack.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/where.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantize_model",
        "//machina/compiler/mlir/lite/quantization/lite:test_util",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_utils",
        "//machina/lite/testing:util",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tflite_portable_test_suite()
