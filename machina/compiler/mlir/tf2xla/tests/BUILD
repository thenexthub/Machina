load("//machina:machina.default.bzl", "filegroup")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "@toolchain-project//mlir:run_lit.sh",
    size_override = {
        "legalize-tf-binary-elementwise.mlir": "medium",
        "legalize-tf-include-tf2xla-fallback.mlir": "medium",
        "legalize-tf-prefer-tf2xla.mlir": "medium",
        "legalize-tf.mlir": "medium",
    },
    test_file_exts = [
        "mlir",
        "hlotxt",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//machina/compiler/mlir:tf-opt",
        "@toolchain-project//toolchain:FileCheck",
    ],
)
