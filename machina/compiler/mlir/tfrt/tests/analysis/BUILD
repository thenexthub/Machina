load("//machina:machina.bzl", "if_oss", "tf_cc_test")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    driver = "//machina/compiler/mlir:run_lit.sh",
    exclude = ["testdata/**"],
    features = if_oss(["--path=org_machina/machina/compiler/mlir/tfrt"]),
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//machina/compiler/mlir:tf-mlir-translate",
        "//machina/compiler/mlir/tfrt:tf-tfrt-opt",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
        "@toolchain-project//mlir:run_lit.sh",
    ],
)

tf_cc_test(
    name = "update_op_cost_in_tfrt_mlir_test",
    srcs = ["update_op_cost_in_tfrt_mlir_test.cc"],
    data = [
        "testdata/test.mlir",
    ],
    deps = [
        "//machina/compiler/mlir/tfrt:transforms/update_op_cost_in_tfrt_mlir",
        "//machina/compiler/mlir/tfrt/ir:tfrt_fallback_async_opdefs",
        "//machina/compiler/mlir/tfrt/ir:tfrt_fallback_sync_opdefs",
        "//machina/core:test",
        "//machina/core/platform:resource_loader",
        "//machina/core/tfrt/fallback:cost_recorder",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@tf_runtime//:init_tfrt_dialects",
    ],
)
