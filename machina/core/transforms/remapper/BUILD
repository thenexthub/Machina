load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library")
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

gentbl_cc_library(
    name = "MklPDLLPatternsIncGen",
    tbl_outs = {"pdll/MklPDLLPatterns.h.inc": ["-x=cpp"]},
    tblgen = "@toolchain-project//mlir:mlir-pdll",
    td_file = "pdll/mkl_patterns.pdll",
    deps = [
        "//machina/core/ir:Dialect",
        "//machina/core/transforms:PDLLUtilsFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "Pass",
    srcs = ["pass.cc"],
    hdrs = [
        "pass.h",
        "remapping_helper.h",
    ],
    deps = [
        ":MklPDLLPatternsIncGen",
        "//machina/core:framework",
        "//machina/core/ir:Dialect",
        "//machina/core/transforms:PassIncGen",
        "//machina/core/transforms:op_cat_helper",
        "//machina/core/transforms:pdll_utils",
        "//machina/core/transforms:utils",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:PDLDialect",
        "@toolchain-project//mlir:PDLInterpDialect",
        "@toolchain-project//mlir:Parser",
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
    test_file_exts = ["mlir"],
)
