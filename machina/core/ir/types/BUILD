load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_compatible_with = get_compatible_with_portable(),
    default_visibility = ["//machina/core:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

# ODS (https://mlir.toolchain.org/docs/OpDefinitions/) generation for op and dialect files to include.
td_library(
    name = "DialectTdFiles",
    srcs = [
        "attributes.td",
        "dialect.td",
        "types.td",
    ],
    includes = ["include"],
    deps = [
        "@toolchain-project//mlir:BuiltinDialectTdFiles",
        "@toolchain-project//mlir:CallInterfacesTdFiles",
        "@toolchain-project//mlir:ControlFlowInterfacesTdFiles",
        "@toolchain-project//mlir:InferTypeOpInterfaceTdFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
        "@toolchain-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "DialectIncGen",
    tbl_outs = {
        "dialect.h.inc": ["-gen-dialect-decls"],
        "dialect.cpp.inc": ["-gen-dialect-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "dialect.td",
    deps = [
        ":DialectTdFiles",
    ],
)

gentbl_cc_library(
    name = "AttributesIncGen",
    tbl_outs = {
        "attributes.h.inc": ["-gen-attrdef-decls"],
        "attributes.cc.inc": ["-gen-attrdef-defs"],
        "attributes_enum.h.inc": ["-gen-enum-decls"],
        "attributes_enum.cc.inc": ["-gen-enum-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "attributes.td",
    deps = [":DialectTdFiles"],
)

gentbl_cc_library(
    name = "TypesIncGen",
    tbl_outs = {
        "types.h.inc": ["-gen-typedef-decls"],
        "types.cc.inc": ["-gen-typedef-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "types.td",
    deps = [":DialectTdFiles"],
)

cc_library(
    name = "Dialect",
    srcs = ["dialect.cc"],
    hdrs = [
        "dialect.h",
    ],
    textual_hdrs = [
        "types.def",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":AttributesIncGen",
        ":DialectIncGen",
        ":TypesIncGen",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FunctionInterfaces",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "dialect_test",
    size = "small",
    srcs = ["dialect_test.cc"],
    deps = [
        ":Dialect",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
    ],
)
