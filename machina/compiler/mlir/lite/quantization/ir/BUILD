load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

td_library(
    name = "QuantizationOpsTdFiles",
    srcs = [
        "QuantOps.td",
        "QuantOpsBase.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@toolchain-project//mlir:InferTypeOpInterfaceTdFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
        "@toolchain-project//mlir:QuantizationOpsTdFiles",
        "@toolchain-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "QuantOpsIncGen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "QuantOps.h.inc": ["-gen-op-decls"],
        "QuantOps.cc.inc": ["-gen-op-defs"],
        "QuantOpsDialect.h.inc": [
            "-gen-dialect-decls",
            "-dialect=quantfork",
        ],
        "QuantOpsDialect.cc.inc": [
            "-gen-dialect-defs",
            "-dialect=quantfork",
        ],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "QuantOps.td",
    deps = [":QuantizationOpsTdFiles"],
)

gentbl_cc_library(
    name = "QuantPassIncGen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"Passes.h.inc": [
        "-gen-pass-decls",
        "-name=quantfork",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    deps = ["@toolchain-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "QuantOps",
    srcs = [
        "ConvertConst.cc",
        "ConvertSimQuant.cc",
        "QuantOps.cc",
        "QuantizeUtils.cc",
    ],
    hdrs = [
        "Passes.h",
        "QuantOps.h",
        "QuantizeUtils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":QuantOpsIncGen",
        ":QuantPassIncGen",
        "//machina/compiler/mlir/quantization/common/ir:QuantOps",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:BytecodeOpInterface",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)
