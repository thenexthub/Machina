load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = internal_visibility(["//learning/brain/mlir:xla_friends"]),
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes.h.inc": [
        "-gen-pass-decls",
        "-name=XlaFramework",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [
        "//xla/mlir_hlo:hlo_ops_td_files",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncTdFiles",
        "@toolchain-project//mlir:PassBaseTdFiles",
        "@toolchain-project//mlir:TensorOpsTdFiles",
    ],
)

cc_library(
    name = "passes",
    srcs = [
        "outline_with_xla_framework.cc",
        "xla_framework_to_llvm_pass.cc",
    ],
    hdrs = [
        "passes.h",
    ],
    deps = [
        ":passes_inc_gen",
        "//xla/mlir/framework/ir:xla_framework",
        "@toolchain-project//toolchain:Core",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LLVMCommonConversion",
        "@toolchain-project//mlir:LLVMDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)
