load("//machina:machina.default.bzl", "filegroup")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/compiler/mlir/lite/stablehlo/odml_converter:__subpackages__"],
)

glob_lit_tests(
    name = "filecheck_tests",
    data = [":test_utilities"],
    driver = "@toolchain-project//mlir:run_lit.sh",
    test_file_exts = [
        "mlir",
    ],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//machina/compiler/mlir/lite/stablehlo/odml_converter:odml-converter",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//mlir:run_lit.sh",
    ],
)
