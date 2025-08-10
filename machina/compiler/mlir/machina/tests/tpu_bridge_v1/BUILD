load("//machina:machina.default.bzl", "filegroup")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

licenses(["notice"])

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "@toolchain-project//mlir:run_lit.sh",
    exclude = [
        # TODO(b/305738491): Temporarily excluded until Bridge V1 has all passes as pass pipelines.
        "end_to_end.mlir",
    ],
    test_file_exts = [
        "mlir",
        "pbtxt",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//machina/compiler/mlir:tf-opt",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
    ],
)
