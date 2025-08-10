load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# LINT.IfChange(cc_library_padding)
cc_library(
    name = "padding",
    srcs = [],
    hdrs = ["padding.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//machina/compiler/mlir/lite/kernels:__pkg__",
        "//machina/compiler/mlir/quantization/machina/utils:__pkg__",
    ],
    deps = [
        "//machina/compiler/mlir/lite/core/c:tflite_common",
    ],
)
# LINT.ThenChange(//machina/lite/kernels/BUILD)
