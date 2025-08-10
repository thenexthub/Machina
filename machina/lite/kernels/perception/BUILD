load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "perception_ops",
    srcs = [
        "dense_image_warp.cc",
        "max_pool_with_argmax.cc",
        "max_unpooling_2d.cc",
        "perception_ops.cc",
    ],
    hdrs = [
        "perception_ops.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/lite:framework",
        "//machina/lite/c:common",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels:padding",
        "//machina/lite/kernels/internal:common",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/kernels/internal:tensor",
        "//machina/lite/kernels/internal:types",
        "@flatbuffers",
    ],
)

cc_test(
    name = "perception_ops_test",
    size = "small",
    srcs = [
        "dense_image_warp_test.cc",
        "max_pool_with_argmax_test.cc",
        "max_unpooling_2d_test.cc",
    ],
    deps = [
        ":perception_ops",
        "//machina/lite:framework",
        "//machina/lite/core:framework_stable",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_main",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

pybind_extension(
    name = "pywrap_perception_ops",
    srcs = [
        "perception_ops_wrapper.cc",
    ],
    hdrs = ["perception_ops.h"],
    additional_exported_symbols = ["PerceptionOpsRegisterer"],
    enable_stub_generation = True,
    link_in_framework = True,
    pytype_srcs = [
        "pywrap_perception_ops.pyi",
    ],
    deps = [
        ":perception_ops",
        "//machina/lite:mutable_op_resolver",
        "//machina/lite/c:common",
        "@pybind11",
    ],
)
