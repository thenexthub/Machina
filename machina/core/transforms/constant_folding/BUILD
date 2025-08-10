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
    name = "Pass",
    srcs = ["pass.cc"],
    hdrs = ["pass.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core/framework:op_def_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/ir:Dialect",
        "//machina/core/ir/importexport:convert_types",
        "//machina/core/ir/types:Dialect",
        "//machina/core/platform:logging",
        "//machina/core/platform:status",
        "//machina/core/transforms:PassIncGen",
        "//machina/core/transforms:eval_utils",
        "//machina/core/transforms:op_cat_helper",
        "//machina/core/transforms:utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Rewrite",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = ["//machina/core/transforms:test_utilities"],
    driver = "@toolchain-project//mlir:run_lit.sh",
    exclude = [],
    test_file_exts = ["mlir"],
)
