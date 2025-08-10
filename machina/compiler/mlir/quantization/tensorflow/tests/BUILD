load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/compiler/mlir/quantization/machina:internal_visibility_allowlist_package",
    ],
    licenses = ["notice"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//machina/compiler/mlir/quantization/machina:tf-quant-opt",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
        "@toolchain-project//mlir:run_lit.sh",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "@toolchain-project//mlir:run_lit.sh",
    size_override = {
        "insert_quantized_functions.mlir": "medium",
        "insert_quantized_functions_drq.mlir": "medium",
        "replace_cast_hacks_with_tf_xla_ops_large_constants.mlir": "medium",
    },
    tags_override = {
        # This test has high memory usage. It causes OOM in msan mode.
        "replace_cast_hacks_with_tf_xla_ops_large_constants.mlir": ["nomsan"],
    },
    test_file_exts = ["mlir"],
)
