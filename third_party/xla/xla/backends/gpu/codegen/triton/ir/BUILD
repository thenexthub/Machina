load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

td_library(
    name = "triton_xla_td_files",
    srcs = glob(["*.td"]),
    includes = ["."],
    deps = [
        "@toolchain-project//mlir:BuiltinDialectTdFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
        "@toolchain-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "triton_xla_dialect_inc_gen",
    strip_include_prefix = ".",
    tbl_outs = {
        "triton_xla_dialect.h.inc": ["-gen-dialect-decls"],
        "triton_xla_dialect.cc.inc": ["-gen-dialect-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "triton_xla_dialect.td",
    deps = [":triton_xla_td_files"],
)

gentbl_cc_library(
    name = "triton_xla_ops_inc_gen",
    strip_include_prefix = ".",
    tbl_outs = {
        "triton_xla_ops.h.inc": ["-gen-op-decls"],
        "triton_xla_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "triton_xla_ops.td",
    deps = [
        ":triton_xla_td_files",
        "@toolchain-project//mlir:InferTypeOpInterfaceTdFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
        "@toolchain-project//mlir:SideEffectInterfacesTdFiles",
        "@triton//:td_files",
    ],
)

gentbl_cc_library(
    name = "triton_xla_attrs_inc_gen",
    strip_include_prefix = ".",
    tbl_outs = {
        "triton_xla_enums.h.inc": ["-gen-enum-decls"],
        "triton_xla_enums.cc.inc": ["-gen-enum-defs"],
        "triton_xla_attrs.h.inc": [
            "-gen-attrdef-decls",
            "-attrdefs-dialect=triton_xla",
        ],
        "triton_xla_attrs.cc.inc": [
            "-gen-attrdef-defs",
            "-attrdefs-dialect=triton_xla",
        ],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "triton_xla_attrs.td",
    deps = [
        ":triton_xla_td_files",
        "@triton//:td_files",
    ],
)

cc_library(
    name = "triton_xla",
    srcs = [
        "triton_xla_attrs.cc",
        "triton_xla_dialect.cc",
        "triton_xla_ops.cc",
    ],
    hdrs = ["triton_xla_ops.h"],
    deps = [
        ":triton_xla_attrs_inc_gen",
        ":triton_xla_dialect_inc_gen",
        ":triton_xla_ops_inc_gen",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:DialectUtils",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:ViewLikeInterface",
        "@triton//:TritonDialects",
    ],
)
