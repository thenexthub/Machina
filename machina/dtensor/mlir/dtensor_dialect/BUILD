load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

# DTensor MLIR dialect.
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/dtensor:dtensor-internal",
        # Allow visibility from the mlir language server.
        "//learning/brain/mlir/mlir_lsp_server:__pkg__",
    ],
    licenses = ["notice"],
)

# ODS (https://mlir.toolchain.org/docs/OpDefinitions/) generation for op and dialect files to include.

td_library(
    name = "dtensor_td_files",
    srcs = [
        "ir/dtensor_dialect.td",
        "ir/dtensor_ops.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@toolchain-project//mlir:FuncTdFiles",
        "@toolchain-project//mlir:InferTypeOpInterfaceTdFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "DialectIncGen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "ir/ops.h.inc": ["-gen-op-decls"],
        "ir/ops.cc.inc": ["-gen-op-defs"],
        "ir/dialect.h.inc": ["-gen-dialect-decls"],
        "ir/dialect.cc.inc": ["-gen-dialect-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "ir/dtensor_ops.td",
    deps = [":dtensor_td_files"],
)

cc_library(
    name = "Dialect",
    srcs = ["ir/ops.cc"],
    hdrs = [
        "ir/dialect.h",
        "ir/ops.h",
    ],
    deps = [
        ":DialectIncGen",
        ":ir/dtensor_attributes",
        "//machina/dtensor/cc:dstatus",
        "//machina/dtensor/cc:tensor_layout",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "ir/dtensor_attributes",
    srcs = ["ir/dtensor_attributes.cc"],
    hdrs = [
        "ir/dialect.h",
        "ir/dtensor_attributes.h",
    ],
    deps = [
        ":DialectIncGen",
        "//machina/dtensor/cc:tensor_layout",
        "//machina/dtensor/proto:layout_proto_cc",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)
