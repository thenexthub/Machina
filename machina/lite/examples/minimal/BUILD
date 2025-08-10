# Description:
#   TensorFlow Lite minimal example.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("//machina/lite:build_def.bzl", "tflite_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_binary(
    name = "minimal",
    srcs = [
        "minimal.cc",
    ],
    linkopts = tflite_linkopts() + select({
        "//machina:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//machina/lite:framework",
        "//machina/lite/core:cc_api_stable",
        "//machina/lite/kernels:builtin_ops",
    ],
)
