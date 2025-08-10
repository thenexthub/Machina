load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    # By default, these targets should only be used within the quantization library.
    default_visibility = [
        "//learning/brain/mlir/quantization:__subpackages__",
        "//machina/compiler/mlir/lite:__subpackages__",
        "//machina/compiler/mlir/quantization:__subpackages__",
        "//machina/compiler/mlir/stablehlo:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "test_base",
    testonly = 1,
    srcs = [],
    hdrs = ["test_base.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/quantization/ir:QuantOps",
        "//machina/compiler/mlir/quantization/common:func",
        "//machina/compiler/mlir/quantization/common/ir:QuantOps",
        "//machina/compiler/mlir/quantization/stablehlo/cc:context",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/core:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)
