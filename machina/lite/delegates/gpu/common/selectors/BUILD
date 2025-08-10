load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "if_google")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

_selectors_package = (
    "//machina/lite/delegates/gpu/common/selectors/" +
    if_google("google", "default")
)

cc_library(
    name = "convolution_selector",
    hdrs = ["convolution_selector.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:model_hints",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:weights_layout",
        _selectors_package + ":convolution_selector",
    ],
)

cc_library(
    name = "convolution_transposed_selector",
    hdrs = ["convolution_transposed_selector.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:weights_layout",
        "@com_google_absl//absl/memory",
        _selectors_package + ":convolution_transposed_selector",
    ],
)

cc_library(
    name = "default_selector",
    hdrs = ["default_selector.h"],
    deps = [
        ":subgraph",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:model_hints",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        _selectors_package + ":default_selector",
    ],
)

cc_library(
    name = "dw_convolution_selector",
    hdrs = ["dw_convolution_selector.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/memory",
        _selectors_package + ":dw_convolution_selector",
    ],
)

cc_library(
    name = "fully_connected_selector",
    hdrs = ["fully_connected_selector.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "@com_google_absl//absl/memory",
        _selectors_package + ":fully_connected_selector",
    ],
)

cc_library(
    name = "operation_selector",
    srcs = ["operation_selector.cc"],
    hdrs = ["operation_selector.h"],
    deps = [
        ":convolution_selector",
        ":convolution_transposed_selector",
        ":default_selector",
        ":dw_convolution_selector",
        ":fully_connected_selector",
        ":simple_selectors",
        ":subgraph",
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:flops_util",
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:model_hints",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/common:winograd_util",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:weights_conversion",
        "//machina/lite/delegates/gpu/common/task:weights_layout",
        "//machina/lite/delegates/gpu/common/tasks:elementwise",
        "//machina/lite/delegates/gpu/common/tasks:mean_stddev_normalization",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "simple_selectors",
    srcs = ["simple_selectors.cc"],
    hdrs = ["simple_selectors.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/tasks:add",
        "//machina/lite/delegates/gpu/common/tasks:cast",
        "//machina/lite/delegates/gpu/common/tasks:concat_xy",
        "//machina/lite/delegates/gpu/common/tasks:concat_z",
        "//machina/lite/delegates/gpu/common/tasks:cumsum",
        "//machina/lite/delegates/gpu/common/tasks:depthwise_conv",
        "//machina/lite/delegates/gpu/common/tasks:gather",
        "//machina/lite/delegates/gpu/common/tasks:lstm",
        "//machina/lite/delegates/gpu/common/tasks:max_unpooling",
        "//machina/lite/delegates/gpu/common/tasks:one_hot",
        "//machina/lite/delegates/gpu/common/tasks:padding",
        "//machina/lite/delegates/gpu/common/tasks:pooling",
        "//machina/lite/delegates/gpu/common/tasks:prelu",
        "//machina/lite/delegates/gpu/common/tasks:quantize_and_dequantize",
        "//machina/lite/delegates/gpu/common/tasks:reduce",
        "//machina/lite/delegates/gpu/common/tasks:relu",
        "//machina/lite/delegates/gpu/common/tasks:resampler",
        "//machina/lite/delegates/gpu/common/tasks:reshape",
        "//machina/lite/delegates/gpu/common/tasks:reshapex4",
        "//machina/lite/delegates/gpu/common/tasks:resize",
        "//machina/lite/delegates/gpu/common/tasks:select_v2",
        "//machina/lite/delegates/gpu/common/tasks:softmax",
        "//machina/lite/delegates/gpu/common/tasks:softmax1x1",
        "//machina/lite/delegates/gpu/common/tasks:space_to_depth",
        "//machina/lite/delegates/gpu/common/tasks:split",
        "//machina/lite/delegates/gpu/common/tasks:strided_slice",
        "//machina/lite/delegates/gpu/common/tasks:tile",
        "//machina/lite/delegates/gpu/common/tasks:transpose",
        "//machina/lite/delegates/gpu/common/tasks:winograd",
    ],
)

cc_library(
    name = "special_selector",
    srcs = ["special_selector.cc"],
    hdrs = ["special_selector.h"],
    deps = [
        ":subgraph",
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:model_hints",
        "//machina/lite/delegates/gpu/common:precision",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/tasks:mean_stddev_normalization",
        "//machina/lite/delegates/gpu/common/tasks/special:conv_pointwise",
        "//machina/lite/delegates/gpu/common/tasks/special:dw7x7_conv2to6_concat_conv8to8",
        "//machina/lite/delegates/gpu/common/tasks/special:fc_fc_add",
        "//machina/lite/delegates/gpu/common/tasks/special:thin_pointwise_fuser",
    ],
)

cc_library(
    name = "subgraph",
    srcs = ["subgraph.cc"],
    hdrs = ["subgraph.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
    ],
)
