load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "add",
    srcs = ["add.cc"],
    hdrs = ["add.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "add_test_util",
    testonly = 1,
    srcs = ["add_test_util.cc"],
    hdrs = ["add_test_util.h"],
    deps = [
        ":add",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "cast",
    srcs = ["cast.cc"],
    hdrs = ["cast.h"],
    deps = [
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "cast_test_util",
    testonly = 1,
    srcs = ["cast_test_util.cc"],
    hdrs = ["cast_test_util.h"],
    deps = [
        ":cast",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "concat_xy",
    srcs = ["concat_xy.cc"],
    hdrs = ["concat_xy.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "concat_z",
    srcs = ["concat_z.cc"],
    hdrs = ["concat_z.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "concat_test_util",
    testonly = 1,
    srcs = ["concat_test_util.cc"],
    hdrs = ["concat_test_util.h"],
    deps = [
        ":concat_xy",
        ":concat_z",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "conv_constants",
    srcs = ["conv_constants.cc"],
    hdrs = ["conv_constants.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:buffer_desc",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_constants_test_util",
    testonly = 1,
    srcs = ["conv_constants_test_util.cc"],
    hdrs = ["conv_constants_test_util.h"],
    deps = [
        ":conv_constants",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "conv_metal_simd",
    srcs = ["conv_metal_simd.cc"],
    hdrs = ["conv_metal_simd.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:weights_layout",
    ],
)

cc_library(
    name = "conv_generic",
    srcs = ["conv_generic.cc"],
    hdrs = ["conv_generic.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:winograd_util",
        "//machina/lite/delegates/gpu/common/task:buffer_desc",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:util",
        "//machina/lite/delegates/gpu/common/task:weights_conversion",
        "//machina/lite/delegates/gpu/common/task:weights_layout",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "conv_generic_test_util",
    testonly = 1,
    srcs = ["conv_generic_test_util.cc"],
    hdrs = ["conv_generic_test_util.h"],
    deps = [
        ":conv_generic",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "conv_weights_converter",
    srcs = ["conv_weights_converter.cc"],
    hdrs = ["conv_weights_converter.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:util",
        "//machina/lite/delegates/gpu/common/task:weights_layout",
    ],
)

cc_library(
    name = "conv_weights_converter_test_util",
    testonly = 1,
    srcs = ["conv_weights_converter_test_util.cc"],
    hdrs = ["conv_weights_converter_test_util.h"],
    deps = [
        ":conv_weights_converter",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
        "//machina/lite/delegates/gpu/common/task:weights_conversion",
    ],
)

cc_library(
    name = "conversion",
    srcs = ["conversion.cc"],
    hdrs = ["conversion.h"],
    deps = [
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "convolution_transposed",
    srcs = ["convolution_transposed.cc"],
    hdrs = ["convolution_transposed.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:buffer_desc",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:weights_conversion",
        "//machina/lite/delegates/gpu/common/task:weights_layout",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "convolution_transposed_test_util",
    testonly = 1,
    srcs = ["convolution_transposed_test_util.cc"],
    hdrs = ["convolution_transposed_test_util.h"],
    deps = [
        ":convolution_transposed",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "convolution_transposed_3x3",
    srcs = ["convolution_transposed_3x3.cc"],
    hdrs = ["convolution_transposed_3x3.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:buffer_desc",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:weights_conversion",
        "//machina/lite/delegates/gpu/common/task:weights_layout",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "convolution_transposed_3x3_test_util",
    testonly = 1,
    srcs = ["convolution_transposed_3x3_test_util.cc"],
    hdrs = ["convolution_transposed_3x3_test_util.h"],
    deps = [
        ":convolution_transposed_3x3",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "convolution_transposed_3x3_thin",
    srcs = ["convolution_transposed_3x3_thin.cc"],
    hdrs = ["convolution_transposed_3x3_thin.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:precision",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:buffer_desc",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:weights_conversion",
        "//machina/lite/delegates/gpu/common/task:weights_layout",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "convolution_transposed_3x3_thin_test_util",
    testonly = 1,
    srcs = ["convolution_transposed_3x3_thin_test_util.cc"],
    hdrs = ["convolution_transposed_3x3_thin_test_util.h"],
    deps = [
        ":convolution_transposed_3x3_thin",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "convolution_transposed_4x4",
    srcs = ["convolution_transposed_4x4.cc"],
    hdrs = ["convolution_transposed_4x4.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:buffer_desc",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:weights_conversion",
        "//machina/lite/delegates/gpu/common/task:weights_layout",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "convolution_transposed_4x4_test_util",
    testonly = 1,
    srcs = ["convolution_transposed_4x4_test_util.cc"],
    hdrs = ["convolution_transposed_4x4_test_util.h"],
    deps = [
        ":convolution_transposed_4x4",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "convolution_transposed_thin",
    srcs = ["convolution_transposed_thin.cc"],
    hdrs = ["convolution_transposed_thin.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:buffer_desc",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "convolution_transposed_thin_test_util",
    testonly = 1,
    srcs = ["convolution_transposed_thin_test_util.cc"],
    hdrs = ["convolution_transposed_thin_test_util.h"],
    deps = [
        ":convolution_transposed_thin",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "cumsum",
    srcs = ["cumsum.cc"],
    hdrs = ["cumsum.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
    ],
)

cc_library(
    name = "cumsum_test_util",
    testonly = 1,
    srcs = ["cumsum_test_util.cc"],
    hdrs = ["cumsum_test_util.h"],
    deps = [
        ":cumsum",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "depthwise_conv",
    srcs = ["depthwise_conv.cc"],
    hdrs = ["depthwise_conv.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/common/task:buffer_desc",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "depthwise_conv_test_util",
    testonly = 1,
    srcs = ["depthwise_conv_test_util.cc"],
    hdrs = ["depthwise_conv_test_util.h"],
    deps = [
        ":depthwise_conv",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "depthwise_conv_3x3",
    srcs = ["depthwise_conv_3x3.cc"],
    hdrs = ["depthwise_conv_3x3.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:buffer_desc",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "depthwise_conv_3x3_test_util",
    testonly = 1,
    srcs = ["depthwise_conv_3x3_test_util.cc"],
    hdrs = ["depthwise_conv_3x3_test_util.h"],
    deps = [
        ":depthwise_conv_3x3",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "depthwise_conv_3x3_stride_h2",
    srcs = ["depthwise_conv_3x3_stride_h2.cc"],
    hdrs = ["depthwise_conv_3x3_stride_h2.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:buffer_desc",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "depthwise_conv_3x3_stride_h2_test_util",
    testonly = 1,
    srcs = ["depthwise_conv_3x3_stride_h2_test_util.cc"],
    hdrs = ["depthwise_conv_3x3_stride_h2_test_util.h"],
    deps = [
        ":depthwise_conv_3x3_stride_h2",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "elementwise",
    srcs = ["elementwise.cc"],
    hdrs = ["elementwise.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:precision",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "elementwise_test_util",
    testonly = 1,
    srcs = ["elementwise_test_util.cc"],
    hdrs = ["elementwise_test_util.h"],
    deps = [
        ":elementwise",
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:precision",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "fully_connected",
    srcs = ["fully_connected.cc"],
    hdrs = ["fully_connected.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:kernel_info",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:precision",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/common/task:buffer_desc",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:tuning_type",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "fully_connected_test_util",
    testonly = 1,
    srcs = ["fully_connected_test_util.cc"],
    hdrs = ["fully_connected_test_util.h"],
    deps = [
        ":fully_connected",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:precision",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "gather",
    srcs = ["gather.cc"],
    hdrs = ["gather.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
    ],
)

cc_library(
    name = "gather_test_util",
    testonly = 1,
    srcs = ["gather_test_util.cc"],
    hdrs = ["gather_test_util.h"],
    deps = [
        ":gather",
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:precision",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "lstm",
    srcs = ["lstm.cc"],
    hdrs = ["lstm.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "lstm_test_util",
    testonly = 1,
    srcs = ["lstm_test_util.cc"],
    hdrs = ["lstm_test_util.h"],
    deps = [
        ":lstm",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "max_unpooling",
    srcs = ["max_unpooling.cc"],
    hdrs = ["max_unpooling.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
    ],
)

cc_library(
    name = "max_unpooling_test_util",
    testonly = 1,
    srcs = ["max_unpooling_test_util.cc"],
    hdrs = ["max_unpooling_test_util.h"],
    deps = [
        ":max_unpooling",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "mean_stddev_normalization",
    srcs = ["mean_stddev_normalization.cc"],
    hdrs = ["mean_stddev_normalization.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/common/selectors:subgraph",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "mean_stddev_normalization_test_util",
    testonly = 1,
    srcs = ["mean_stddev_normalization_test_util.cc"],
    hdrs = ["mean_stddev_normalization_test_util.h"],
    deps = [
        ":mean_stddev_normalization",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "one_hot",
    srcs = ["one_hot.cc"],
    hdrs = ["one_hot.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "one_hot_test_util",
    testonly = 1,
    srcs = ["one_hot_test_util.cc"],
    hdrs = ["one_hot_test_util.h"],
    deps = [
        ":one_hot",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "padding",
    srcs = ["padding.cc"],
    hdrs = ["padding.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "padding_test_util",
    testonly = 1,
    srcs = ["padding_test_util.cc"],
    hdrs = ["padding_test_util.h"],
    deps = [
        ":padding",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "pooling",
    srcs = ["pooling.cc"],
    hdrs = ["pooling.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:util",
    ],
)

cc_library(
    name = "pooling_test_util",
    testonly = 1,
    srcs = ["pooling_test_util.cc"],
    hdrs = ["pooling_test_util.h"],
    deps = [
        ":pooling",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "prelu",
    srcs = ["prelu.cc"],
    hdrs = ["prelu.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "prelu_test_util",
    testonly = 1,
    srcs = ["prelu_test_util.cc"],
    hdrs = ["prelu_test_util.h"],
    deps = [
        ":prelu",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "quantize_and_dequantize",
    srcs = ["quantize_and_dequantize.cc"],
    hdrs = ["quantize_and_dequantize.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "quantize_and_dequantize_test_util",
    testonly = 1,
    srcs = ["quantize_and_dequantize_test_util.cc"],
    hdrs = ["quantize_and_dequantize_test_util.h"],
    deps = [
        ":quantize_and_dequantize",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
        "//machina/lite/kernels/internal:quantization_util",
    ],
)

cc_library(
    name = "reduce",
    srcs = ["reduce.cc"],
    hdrs = ["reduce.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:kernel_info",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:util",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "reduce_test_util",
    testonly = 1,
    srcs = ["reduce_test_util.cc"],
    hdrs = ["reduce_test_util.h"],
    deps = [
        ":reduce",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "relu",
    srcs = ["relu.cc"],
    hdrs = ["relu.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "relu_test_util",
    testonly = 1,
    srcs = ["relu_test_util.cc"],
    hdrs = ["relu_test_util.h"],
    deps = [
        ":relu",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "resampler",
    srcs = ["resampler.cc"],
    hdrs = ["resampler.h"],
    deps = [
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
    ],
)

cc_library(
    name = "resampler_test_util",
    testonly = 1,
    srcs = ["resampler_test_util.cc"],
    hdrs = ["resampler_test_util.h"],
    deps = [
        ":resampler",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "reshape",
    srcs = ["reshape.cc"],
    hdrs = ["reshape.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "reshapex4",
    srcs = ["reshapex4.cc"],
    hdrs = ["reshapex4.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "reshape_test_util",
    testonly = 1,
    srcs = ["reshape_test_util.cc"],
    hdrs = ["reshape_test_util.h"],
    deps = [
        ":reshape",
        ":reshapex4",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "resize",
    srcs = ["resize.cc"],
    hdrs = ["resize.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "resize_test_util",
    testonly = 1,
    srcs = ["resize_test_util.cc"],
    hdrs = ["resize_test_util.h"],
    deps = [
        ":resize",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "select_v2",
    srcs = ["select_v2.cc"],
    hdrs = ["select_v2.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "select_v2_test_util",
    testonly = 1,
    srcs = ["select_v2_test_util.cc"],
    hdrs = ["select_v2_test_util.h"],
    deps = [
        ":select_v2",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common/task:testing_util",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "softmax",
    srcs = ["softmax.cc"],
    hdrs = ["softmax.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "softmax1x1",
    srcs = ["softmax1x1.cc"],
    hdrs = ["softmax1x1.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:util",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "softmax_test_util",
    testonly = 1,
    srcs = ["softmax_test_util.cc"],
    hdrs = ["softmax_test_util.h"],
    deps = [
        ":softmax",
        ":softmax1x1",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "space_to_depth",
    srcs = ["space_to_depth.cc"],
    hdrs = ["space_to_depth.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "space_to_depth_test_util",
    testonly = 1,
    srcs = ["space_to_depth_test_util.cc"],
    hdrs = ["space_to_depth_test_util.h"],
    deps = [
        ":space_to_depth",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "split",
    srcs = ["split.cc"],
    hdrs = ["split.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
    ],
)

cc_library(
    name = "split_test_util",
    testonly = 1,
    srcs = ["split_test_util.cc"],
    hdrs = ["split_test_util.h"],
    deps = [
        ":split",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "strided_slice",
    srcs = ["strided_slice.cc"],
    hdrs = ["strided_slice.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
    ],
)

cc_library(
    name = "strided_slice_test_util",
    testonly = 1,
    srcs = ["strided_slice_test_util.cc"],
    hdrs = ["strided_slice_test_util.h"],
    deps = [
        ":strided_slice",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "tile",
    srcs = ["tile.cc"],
    hdrs = ["tile.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
    ],
)

cc_library(
    name = "tile_test_util",
    testonly = 1,
    srcs = ["tile_test_util.cc"],
    hdrs = ["tile_test_util.h"],
    deps = [
        ":tile",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "transpose",
    srcs = ["transpose.cc"],
    hdrs = ["transpose.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "transpose_test_util",
    testonly = 1,
    srcs = ["transpose_test_util.cc"],
    hdrs = ["transpose_test_util.h"],
    deps = [
        ":transpose",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)

cc_library(
    name = "winograd",
    srcs = ["winograd.cc"],
    hdrs = ["winograd.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:winograd_util",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:work_group_picking",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "winograd_test_util",
    testonly = 1,
    srcs = ["winograd_test_util.cc"],
    hdrs = ["winograd_test_util.h"],
    deps = [
        ":winograd",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:winograd_util",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ],
)
