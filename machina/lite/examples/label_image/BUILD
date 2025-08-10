# Description:
# TensorFlow Lite Example Label Image.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina/lite:build_def.bzl", "tflite_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(glob([
    "testdata/*.bmp",
]))

cc_binary(
    name = "label_image",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "label_image.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//machina:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-Wl,-rpath=/data/local/tmp",  # for hexagon delegate
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":bitmap_helpers",
        "//machina/lite:framework",
        "//machina/lite:string",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core:cc_api_stable",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/profiling:profile_buffer",
        "//machina/lite/profiling:profiler",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/tools:command_line_flags",
        "//machina/lite/tools:tool_params",
        "//machina/lite/tools/delegates:delegate_provider_hdr",
        "//machina/lite/tools/delegates:tflite_execution_providers",
    ],
)

cc_library(
    name = "bitmap_helpers",
    srcs = ["bitmap_helpers.cc"],
    hdrs = [
        "bitmap_helpers.h",
        "bitmap_helpers_impl.h",
        "label_image.h",
        "log.h",
    ],
    deps = [
        "//machina/core/platform:tstring",
        "//machina/lite:builtin_op_data",
        "//machina/lite:framework",
        "//machina/lite:string",
        "//machina/lite:string_util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/schema:schema_fbs",
        "@local_tsl//tsl/platform:tstring",
    ] + select({
        "//machina:android": [
            "//machina/lite/delegates/gpu:delegate",
            "//machina/lite/delegates/hexagon:hexagon_delegate",
        ],
        "//machina:android_arm64": [
            "//machina/lite/delegates/gpu:delegate",
            "//machina/lite/delegates/hexagon:hexagon_delegate",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "label_image_test",
    srcs = [
        "get_top_n.h",
        "get_top_n_impl.h",
        "label_image_test.cc",
    ],
    data = [
        "testdata/grace_hopper.bmp",
    ],
    deps = [
        ":bitmap_helpers",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)
