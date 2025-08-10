load("//machina:machina.bzl", "if_oss")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    # Custom driver is unsupported in OSS. Fails if one is provided.
    # copybara:uncomment driver = "//machina/compiler/mlir:run_lit.sh",
    # copybara:comment_begin(JitRt/Auto fusion depreciated)
    exclude = [
        "auto-fusion.mlir",
        "tf_to_corert_pipeline_cpurt.mlir",
        "outline-cpurt-cluster.mlir",
    ],
    # copybara:comment_end
    features = if_oss(["--path=org_machina/machina/compiler/mlir/tfrt"]),
    size_override = {
        "fallback.mlir": "medium",
    },
    test_file_exts = ["mlir"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        "//machina/compiler/mlir/tfrt:tf-tfrt-opt",
        "@toolchain-project//toolchain:FileCheck",
        "@toolchain-project//toolchain:not",
        "@toolchain-project//mlir:run_lit.sh",
    ],
)
