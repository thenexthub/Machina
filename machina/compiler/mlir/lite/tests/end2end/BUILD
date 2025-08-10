load("//machina:machina.default.bzl", "filegroup")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

licenses(["notice"])

glob_lit_tests(
    name = "all_tests",
    data = [
        ":quant_stats_files",
        ":test_utilities",
    ],
    driver = "@toolchain-project//mlir:run_lit.sh",
    size_override = {
        "quant_stats.pbtxt": "medium",
    },
    tags_override = {
        "add.pbtxt": ["no_rocm"],
        "fake_quant_per_channel.pbtxt": ["no_rocm"],
    },
    test_file_exts = [
        "pbtxt",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//machina/compiler/mlir/lite:flatbuffer_to_string",
        "//machina/compiler/mlir/lite:tf_tfl_translate",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
    ],
)

# Bundle together all the quant stats files that are used by the tests.
filegroup(
    name = "quant_stats_files",
    srcs = glob(
        ["**/*.stats"],
    ),
)
