load("//machina:machina.bzl", "tf_native_cc_binary")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_compatible_with = get_compatible_with_portable(),
    default_visibility = [":__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

# Custom `mlir-opt` replacement that links our dialect.
# The only purpose is testing the IR constructs themselves (parser, printer,
# verifiers...)
tf_native_cc_binary(
    name = "tfg-opt-no-passes",
    srcs = ["tfg-opt-no-passes.cc"],
    # Expose the test driver to core/ir.
    deps = [
        "//machina/core/ir:Dialect",
        "//machina/core/ir/types:Dialect",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:Transforms",
    ],
)

# Bundle together all of the test utilities that are used by tests.
exports_files(["run_lit.sh"])

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":tfg-opt-no-passes",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "@toolchain-project//mlir:run_lit.sh",
    exclude = [],
    test_file_exts = ["mlir"],
)
