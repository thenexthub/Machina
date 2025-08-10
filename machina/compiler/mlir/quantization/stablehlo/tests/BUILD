load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "filegroup")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    # TODO: b/288344501 - Enable OSS tests again when stable-quant-opt works well.
    default_tags = [
        "no_oss",
        "no_pip",
    ],
    driver = "//machina/compiler/mlir/quantization/stablehlo:run_lit.sh",
    size_override = {
    },
    tags_override = {
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//machina/compiler/mlir/quantization/stablehlo:stablehlo-quant-opt",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
        "@toolchain-project//mlir:run_lit.sh",
    ],
)

tf_cc_test(
    name = "fill_quantization_options_test",
    srcs = ["fill_quantization_options_test.cc"],
    deps = [
        "//machina/compiler/mlir/quantization/stablehlo:fill_quantization_options",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_options_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@local_tsl//tsl/platform:protobuf",
    ],
)
