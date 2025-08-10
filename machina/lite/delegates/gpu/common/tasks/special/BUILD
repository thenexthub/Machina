load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina/lite/delegates/gpu:build_defs.bzl", "gtest_main_no_heapcheck_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "conv_pointwise",
    srcs = ["conv_pointwise.cc"],
    hdrs = ["conv_pointwise.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/common/selectors:subgraph",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "conv_pointwise_test",
    srcs = ["conv_pointwise_test.cc"],
    tags = [
        "noasan",
        "nomsan",
        "notsan",
        "requires-gpu-nvidia",
    ],
    # TODO(b/279977471) Once b/279347631 is resolved, check for heap again
    deps = [
        ":conv_pointwise",
        "//machina/lite/delegates/gpu/cl/kernels:cl_test",
        "//machina/lite/delegates/gpu/common:precision",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
        "//machina/lite/delegates/gpu/common/task:testing_util",
    ] + gtest_main_no_heapcheck_deps(),
)

cc_library(
    name = "dw7x7_conv2to6_concat_conv8to8",
    srcs = ["dw7x7_conv2to6_concat_conv8to8.cc"],
    hdrs = ["dw7x7_conv2to6_concat_conv8to8.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:flops_util",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:precision",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common/selectors:subgraph",
        "//machina/lite/delegates/gpu/common/task:qcom_thin_filter_desc",
    ],
)

cc_library(
    name = "fc_fc_add",
    srcs = ["fc_fc_add.cc"],
    hdrs = ["fc_fc_add.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:operations",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/common/selectors:subgraph",
        "//machina/lite/delegates/gpu/common/task:buffer_desc",
        "//machina/lite/delegates/gpu/common/task:gpu_operation",
    ],
)

cc_library(
    name = "thin_pointwise_fuser",
    srcs = ["thin_pointwise_fuser.cc"],
    hdrs = ["thin_pointwise_fuser.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:flops_util",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:precision",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/common/selectors:subgraph",
        "//machina/lite/delegates/gpu/common/task:tensor_desc",
        "//machina/lite/delegates/gpu/common/task:util",
        "//machina/lite/delegates/gpu/common/tasks:elementwise",
        "//machina/lite/delegates/gpu/common/tasks:prelu",
        "//machina/lite/delegates/gpu/common/tasks:relu",
        "@com_google_absl//absl/strings",
    ],
)
