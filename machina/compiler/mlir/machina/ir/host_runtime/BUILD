load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/compiler/mlir/machina:__pkg__",
        "//machina/compiler/mlir/tfrt:__subpackages__",
    ],
    licenses = ["notice"],
)

td_library(
    name = "machina_tfrt_ops_td_files",
    srcs = [
        "tfrt_ops.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//mlir:CallInterfacesTdFiles",
        "@toolchain-project//mlir:ControlFlowInterfacesTdFiles",
        "@toolchain-project//mlir:InferTypeOpInterfaceTdFiles",
        "@toolchain-project//mlir:LoopLikeInterfaceTdFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
        "@toolchain-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "machina_tfrt_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "tfrt_ops.h.inc": ["-gen-op-decls"],
        "tfrt_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "tfrt_ops.td",
    deps = [
        ":machina_tfrt_ops_td_files",
    ],
)

cc_library(
    name = "machina_tfrt_ops",
    srcs = [
        "tfrt_ops.cc",
    ],
    hdrs = [
        "tfrt_ops.h",
    ],
    textual_hdrs = [
        "tfrt_ops.h.inc",
    ],
    deps = [
        ":machina_tfrt_ops_inc_gen",
        "//machina/compiler/mlir/machina:side_effect_analysis_util",
        "//machina/compiler/mlir/machina:machina_attributes",
        "//machina/compiler/mlir/machina:machina_op_interfaces",
        "//machina/compiler/mlir/machina:machina_op_interfaces_inc_gen",
        "//machina/compiler/mlir/machina:machina_traits",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/compiler/mlir/machina/transforms:rewrite_util",
        "//machina/compiler/mlir/machina/transforms:machina_canonicalize_inc_gen",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/framework:resource_handle",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:BytecodeOpInterface",
        "@toolchain-project//mlir:ControlFlowInterfaces",
        "@toolchain-project//mlir:DerivedAttributeOpInterface",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:LoopLikeInterface",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
    ],
)
