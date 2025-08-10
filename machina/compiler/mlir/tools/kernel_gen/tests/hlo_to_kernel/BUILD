load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    default_tags = [
        # We need access to the CUDA SDK.
        "gpu",
        "no_rocm",
        "nomsan",
        "notsan",
    ],
    driver = "//machina/compiler/mlir:run_lit.sh",
    hermetic_cuda_data_dir = "%S/../../../../../../../../cuda_nvcc",
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//machina/compiler/mlir/tools/kernel_gen:hlo_to_kernel",
        "@toolchain-project//mlir:run_lit.sh",
    ],
)
