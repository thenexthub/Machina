load("//machina:machina.bzl", "tf_native_cc_binary")
load("//machina:machina.default.bzl", "filegroup")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

licenses(["notice"])

glob_lit_tests(
    name = "all_tests",
    data = [
        ":extra_files",
        ":test_utilities",
    ],
    driver = "@toolchain-project//mlir:run_lit.sh",
    test_file_exts = [
        "mlir",
        "cc",
        "json",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":importer_test_min_max",
        ":test_schema.fbs",
        "//machina/compiler/mlir/lite:flatbuffer_to_string",
        "//machina/compiler/mlir/lite:flatbuffer_translate",
        "//machina/compiler/mlir/lite:json_to_flatbuffer",
        "//machina/compiler/mlir/lite:litert-opt",
        "//machina/compiler/mlir/lite:tf_tfl_translate",
        "@toolchain-project//toolchain:FileCheck",
    ],
)

# Bundle together all the files that are used by the non-mlir file-based tests.
filegroup(
    name = "extra_files",
    srcs = glob(
        [
            "**/importer_test_min_max.cc.mlir",
            "**/reshape.mlir",
        ],
    ),
)

# A binary to inject min/max to a tflite model.
# A file check command is used to verify the imported result from this
# binary format.
tf_native_cc_binary(
    name = "importer_test_min_max",
    srcs = [
        "importer_test_min_max.cc",
    ],
    deps = [
        "//machina/compiler/mlir/lite/core:absl_error_model_builder",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/lite/schema:schema_utils",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
    ],
)
