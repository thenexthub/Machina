load("//machina:machina.default.bzl", "filegroup")
load("//machina/compiler/mlir/tosa:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    default_tags = ["tf_tosa"],
    driver = "@toolchain-project//mlir:run_lit.sh",
    size_override = {
        "tf-to-tosa-pipeline.mlir": "medium",
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//machina/compiler/mlir/tosa:tf-tosa-opt",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
    ],
)
