load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_compatible_with = get_compatible_with_portable(),
    default_visibility = [
        "//machina/core:__subpackages__",
    ],
    licenses = ["notice"],  # apache 2.0
)

cc_library(
    name = "GraphToFunc",
    srcs = ["graph_to_func.cc"],
    hdrs = ["graph_to_func.h"],
    deps = [
        "//machina/core/ir:Dialect",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "Pass",
    srcs = ["pass.cc"],
    hdrs = ["pass.h"],
    deps = [
        ":GraphToFunc",
        "//machina/core/ir:Dialect",
        "//machina/core/transforms:PassIncGen",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = ["//machina/core/transforms:test_utilities"],
    driver = "@toolchain-project//mlir:run_lit.sh",
    exclude = [],
    test_file_exts = ["mlir"],
)
