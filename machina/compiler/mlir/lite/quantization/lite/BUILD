load("//machina:machina.bzl", "tf_cc_binary", "tf_cc_test")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:LICENSE"],
    default_visibility = [
        ":friends",
        "//machina:__pkg__",
    ],
    licenses = ["notice"],
)

exports_files(glob([
    "testdata/*.bin",
]))

package_group(
    name = "friends",
    packages = [
        "//machina/compiler/mlir/lite/...",
        "//machina/lite/...",
    ],
)

cc_library(
    name = "quantize_model",
    srcs = [
        "quantize_model.cc",
    ],
    hdrs = [
        "quantize_model.h",
    ],
    deps = [
        "//machina/compiler/mlir/lite:common",
        "//machina/compiler/mlir/lite:flatbuffer_translate_lib",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite:tf_tfl_passes",
        "//machina/compiler/mlir/lite/debug",
        "//machina/compiler/mlir/lite/debug:debug_options_proto_cc",
        "//machina/compiler/mlir/lite/quantization/common/quantization_lib:quantization_config",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "quantize_weights",
    srcs = [
        "quantize_weights.cc",
    ],
    hdrs = [
        "quantize_weights.h",
    ],
    deps = [
        "//machina/compiler/mlir/lite:common",
        "//machina/compiler/mlir/lite:flatbuffer_translate_lib",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite:tf_tfl_passes",
        "//machina/compiler/mlir/lite/quantization/common/quantization_lib:quantization_config",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers//:runtime_cc",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tfl_to_std",
    srcs = [
        "tfl_to_std.cc",
    ],
    hdrs = [
        "tfl_to_std.h",
        "//machina/compiler/mlir/lite/quantization/common/quantization_lib:quantization_utils.h",
    ],
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/quantization/common/quantization_lib",
        "//machina/compiler/mlir/lite/quantization/common/quantization_lib:quantization_config",
        "//machina/compiler/mlir/lite/quantization/ir:QuantOps",
        "//machina/compiler/mlir/quantization/common/ir:QuantOps",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
    ],
)

# Binary to apply quantization on the annotated files.
tf_cc_binary(
    name = "tfl_quantizer",
    srcs = [
        "tfl_quantizer.cc",
    ],
    deps = [
        ":quantize_model",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
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
        ":test_util",
        "//machina/compiler/mlir/lite/core:absl_error_model_builder",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/lite/schema:schema_utils",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
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
        # TODO(b/327796566): re-enable after the bug is fixed
        "manual",
        "noasan",
        "noguitar",
        "notap",
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":quantize_weights",
        ":test_util",
        "//machina/compiler/mlir/lite/core:absl_error_model_builder",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/lite/schema:schema_utils",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
        "@local_xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/lite/core/api:error_reporter",
        "@com_google_googletest//:gtest",
    ],
)
