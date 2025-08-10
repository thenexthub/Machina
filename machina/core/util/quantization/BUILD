# Description: Utils for Tensorflow quantization.

load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load("//machina:machina.default.bzl", "filegroup")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//machina/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "uniform_quant_ops_params.cc",
        "uniform_quant_ops_params.h",
    ],
)

cc_library(
    name = "uniform_quant_ops_params",
    srcs = [
        "uniform_quant_ops_params.cc",
    ],
    hdrs = [
        "uniform_quant_ops_params.h",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
    ],
)

tf_cc_test(
    name = "uniform_quant_ops_params_test",
    srcs = [
        "uniform_quant_ops_params_test.cc",
    ],
    deps = [
        ":uniform_quant_ops_attr_proto_cc",
        ":uniform_quant_ops_params",
        "//machina/core:framework",
        "//machina/core:test",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "uniform_quant_ops_attr_proto",
    srcs = ["uniform_quant_ops_attr.proto"],
    make_default_target_header_only = True,
)
