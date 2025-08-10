load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["@stablehlo//:license"],
    default_visibility = [
        "//machina/compiler/mlir/quantization/stablehlo:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "save_report",
    srcs = ["save_report.cc"],
    hdrs = ["save_report.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/stablehlo/cc:report",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "save_report_test",
    srcs = ["save_report_test.cc"],
    deps = [
        ":save_report",
        "//machina/compiler/mlir/quantization/common:test_base",
        "//machina/compiler/mlir/quantization/stablehlo:passes",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo/cc:io",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/tsl/platform:status_matchers",
    ],
)
