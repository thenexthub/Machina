load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:private",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "portable_tensor_utils",
    srcs = ["portable_tensor_utils.cc"],
    hdrs = ["portable_tensor_utils.h"],
    visibility = [
        "//machina/compiler/mlir/lite/quantization/common/quantization_lib:__pkg__",
        "//machina/compiler/mlir/quantization/common/quantization_lib:__pkg__",
    ],
)

cc_library(
    name = "model_utils",
    srcs = ["model_utils.cc"],
    hdrs = ["model_utils.h"],
    deps = [
        "//machina/compiler/mlir/lite/schema:schema_conversion_utils",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/lite/schema:schema_utils",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "model_utils_test",
    srcs = ["model_utils_test.cc"],
    deps = [
        ":model_utils",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "quantization_utils",
    srcs = ["quantization_utils.cc"],
    hdrs = ["quantization_utils.h"],
    deps = [
        ":model_utils",
        ":portable_tensor_utils",
        "//machina/compiler/mlir/lite/kernels/internal:runtime_shape",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
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
    deps = [
        ":quantization_utils",
        "//machina/compiler/mlir/lite/core:absl_error_model_builder",
        "//machina/compiler/mlir/lite/quantization/lite:test_util",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/lite/schema:schema_utils",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/util:command_line_flags",
    ],
)

cc_library(
    name = "quantize_weights",
    srcs = select({
        "//machina:ios": ["quantize_weights_portable.cc"],
        "//machina:android": ["quantize_weights_portable.cc"],
        "//conditions:default": ["quantize_weights.cc"],
    }),
    hdrs = ["quantize_weights.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":model_utils",
        ":portable_tensor_utils",
        ":quantization_utils",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/lite/schema:schema_utils",
        "//machina/core/platform:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@flatbuffers//:runtime_cc",
    ] + select({
        "//machina:android": [],
        "//machina:ios": [],
        "//conditions:default": [
            "//machina/compiler/mlir/lite/quantization/lite:quantize_weights",
        ],
    }),
)

tf_cc_test(
    name = "quantize_weights_test",
    srcs = ["quantize_weights_test.cc"],
    args = [
        "--test_model_file=$(location //machina/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin)",
    ],
    data = [
        "//machina/compiler/mlir/lite/quantization/lite:testdata/custom_op.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/quantized_with_gather.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/single_conv_weights_min_0_max_plus_10.bin",
        "//machina/compiler/mlir/lite/quantization/lite:testdata/weight_shared_between_convs.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantize_weights",
        "//machina/compiler/mlir/lite/core:absl_error_model_builder",
        "//machina/compiler/mlir/lite/quantization/lite:test_util",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/lite/schema:schema_utils",
        "//machina/core:framework_internal",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
        "@local_xla//xla/tsl/util:command_line_flags",
    ],
)
