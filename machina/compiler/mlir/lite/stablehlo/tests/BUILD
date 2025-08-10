load("//machina:machina.default.bzl", "filegroup")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:private",
    ],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "//machina/compiler/mlir/lite/stablehlo:run_lit.sh",
    size_override = {
        "legalize-skip-quantization-ops.mlir": "medium",
    },
    test_file_exts = [
        "mlir",
        "cc",
    ],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//machina/compiler/mlir/lite:flatbuffer_translate",
        "//machina/compiler/mlir/lite:tf_tfl_translate",
        "//machina/compiler/mlir/lite/stablehlo:odml-to-stablehlo-opt",
        "//machina/compiler/mlir/lite/stablehlo:odml_to_stablehlo",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//mlir:run_lit.sh",
    ],
)
