load("//machina:machina.bzl", "if_oss")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "//machina/compiler/mlir:run_lit.sh",
    features = if_oss(["--path=org_machina/machina/compiler/mlir/tfrt"]),
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//machina/compiler/mlir:tf-mlir-translate",
        "//machina/compiler/mlir:tf-opt",
        "//machina/compiler/mlir/tfrt:tf-tfrt-opt",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
        "@toolchain-project//mlir:run_lit.sh",
    ],
)
