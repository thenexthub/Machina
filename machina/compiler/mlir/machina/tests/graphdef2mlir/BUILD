load("//machina:machina.default.bzl", "filegroup")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

licenses(["notice"])

glob_lit_tests(
    name = "all_tests",
    data = [
        ":debug_info_files",
        ":test_utilities",
    ],
    driver = "@toolchain-project//mlir:run_lit.sh",
    tags_override = {
        "error-message-with-source-info.pbtxt": ["no_oss"],  # TODO(b/150946057): to be fixed on oss.
    },
    test_file_exts = ["pbtxt"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//machina/compiler/mlir:tf-mlir-translate",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
    ],
)

# Bundle together all the debug info files that are used by the tests.
filegroup(
    name = "debug_info_files",
    srcs = glob(
        [
            "**/*.debug",
        ],
    ),
)
