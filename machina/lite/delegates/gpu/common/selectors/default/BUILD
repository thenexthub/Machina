load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "convolution_selector",
    srcs = ["convolution_selector.cc"],
    deps = [
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:model_hints",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:precision",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:weights_layout",
        "//machina/lite/delegates/gpu/common/tasks:conv_constants",
        "//machina/lite/delegates/gpu/common/tasks:conv_generic",
        "//machina/lite/delegates/gpu/common/tasks:conv_metal_simd",
        "//machina/lite/delegates/gpu/common/tasks:conv_weights_converter",
    ],
)

cc_library(
    name = "convolution_transposed_selector",
    srcs = ["convolution_transposed_selector.cc"],
    deps = [
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:weights_layout",
        "//machina/lite/delegates/gpu/common/tasks:convolution_transposed",
        "//machina/lite/delegates/gpu/common/tasks:convolution_transposed_3x3",
        "//machina/lite/delegates/gpu/common/tasks:convolution_transposed_3x3_thin",
        "//machina/lite/delegates/gpu/common/tasks:convolution_transposed_4x4",
        "//machina/lite/delegates/gpu/common/tasks:convolution_transposed_thin",
    ],
)

cc_library(
    name = "default_selector",
    srcs = ["default_selector.cc"],
    deps = [
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:model_hints",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/selectors:subgraph",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dw_convolution_selector",
    srcs = ["dw_convolution_selector.cc"],
    deps = [
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:precision",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/tasks:depthwise_conv",
        "//machina/lite/delegates/gpu/common/tasks:depthwise_conv_3x3",
        "//machina/lite/delegates/gpu/common/tasks:depthwise_conv_3x3_stride_h2",
    ],
)

cc_library(
    name = "fully_connected_selector",
    srcs = ["fully_connected_selector.cc"],
    deps = [
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/tasks:conv_generic",
        "//machina/lite/delegates/gpu/common/tasks:fully_connected",
    ],
)
