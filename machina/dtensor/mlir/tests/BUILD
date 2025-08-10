load("//machina:machina.bzl", "tf_cc_binary")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "@toolchain-project//mlir:run_lit.sh",
    # TODO(b/177569789): Fix below tests to use V2.
    exclude = [
        "layout_propagation.mlir",
        "spmd_fill.mlir",
        "spmd_metadata.mlir",
        "spmd_reduction.mlir",
        "spmd_tile.mlir",
    ],
    tags_override = {
        "move_compilation_to_host.mlir": ["no_oss"],  # FIXME(b/264922760): The test fails on OSS.
        "spmd_dtensor_ops.mlir": ["no_oss"],  # FIXME(b/264922760): The test fails on OSS.
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":dtensor-opt",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
    ],
)

tf_cc_binary(
    name = "dtensor-opt",
    srcs = ["dtensor_mlir_opt_main.cc"],
    deps = [
        "//machina/compiler/mlir:init_mlir",
        "//machina/compiler/mlir/machina",
        "//machina/core:ops",
        "//machina/dtensor/cc:dtensor_ops",
        "//machina/dtensor/mlir:create_dtensor_mlir_passes",
        "//machina/dtensor/mlir:dtensor_mlir_passes",
        "//machina/dtensor/mlir:tf_dtensor_dialect",
        "//machina/dtensor/mlir/dtensor_dialect:Dialect",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AllExtensions",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:RegisterAllDialects",
        "@toolchain-project//mlir:RegisterAllExtensions",
        "@toolchain-project//mlir:RegisterAllPasses",
        "@stablehlo//:register",
    ],
)
