load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

td_library(
    name = "mlrt_td_files",
    srcs = [
        "mlrt_dialect.td",
        "mlrt_ops.td",
    ],
    includes = ["."],
    visibility = ["//visibility:private"],
    deps = [
        "@toolchain-project//mlir:OpBaseTdFiles",
        "@toolchain-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "mlrt_ops_inc_gen",
    tbl_outs = {
        "mlrt_ops.h.inc": ["-gen-op-decls"],
        "mlrt_ops.cpp.inc": ["-gen-op-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "mlrt_ops.td",
    deps = [":mlrt_td_files"],
)

cc_library(
    name = "mlrt_ops",
    srcs = [
        "mlrt_dialect.cc",
        "mlrt_ops.cc",
    ],
    hdrs = [
        "mlrt_dialect.h",
        "mlrt_ops.h",
    ],
    visibility = ["//machina/compiler/mlir/tfrt:__subpackages__"],
    deps = [
        ":mlrt_ops_inc_gen",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InliningUtils",
        "@toolchain-project//mlir:Support",
    ],
)

td_library(
    name = "tf_mlrt_td_files",
    srcs = [
        "tf_mlrt_dialect.td",
        "tf_mlrt_ops.td",
        "tf_ops.td",
    ],
    includes = ["."],
    deps = [
        ":mlrt_td_files",
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//mlir:OpBaseTdFiles",
        "@toolchain-project//mlir:SideEffectInterfacesTdFiles",
        "@tf_runtime//:compiler_td_files",
    ],
)

td_library(
    name = "tf_mlrt_tpu_td_files",
    srcs = [
        "tf_mlrt_tpu_ops.td",
    ],
    includes = ["."],
    visibility = ["//visibility:private"],
    deps = [
        ":mlrt_td_files",
        ":tf_mlrt_td_files",
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//mlir:OpBaseTdFiles",
        "@toolchain-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "tf_mlrt_ops_inc_gen",
    tbl_outs = {
        "tf_mlrt_ops.h.inc": ["-gen-op-decls"],
        "tf_mlrt_ops.cpp.inc": ["-gen-op-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "tf_mlrt_ops.td",
    deps = [":tf_mlrt_td_files"],
)

gentbl_cc_library(
    name = "tf_mlrt_tpu_ops_inc_gen",
    tbl_outs = {
        "tf_mlrt_tpu_ops.h.inc": ["-gen-op-decls"],
        "tf_mlrt_tpu_ops.cpp.inc": ["-gen-op-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "tf_mlrt_tpu_ops.td",
    deps = [":tf_mlrt_tpu_td_files"],
)

gentbl_cc_library(
    name = "tf_ops_inc_gen",
    tbl_outs = {
        "tf_ops.h.inc": ["-gen-op-decls"],
        "tf_ops.cpp.inc": ["-gen-op-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "tf_ops.td",
    deps = [":tf_mlrt_td_files"],
)

cc_library(
    name = "tf_mlrt_ops",
    srcs = ["tf_mlrt_ops.cc"],
    hdrs = ["tf_mlrt_ops.h"],
    visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt/mlrt/application/machina/tests:__subpackages__",
        "//machina/compiler/mlir/tfrt:__subpackages__",
    ],
    deps = [
        ":mlrt_ops",
        ":tf_mlrt_ops_inc_gen",
        ":tf_ops_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_side_effects",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InliningUtils",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
        "@tf_runtime//:compiler_tfrt_op_interfaces",
        "@tf_runtime//:compiler_tfrt_traits",
    ],
)

cc_library(
    name = "tf_mlrt_tpu_ops",
    srcs = ["tf_mlrt_tpu_ops.cc"],
    hdrs = ["tf_mlrt_tpu_ops.h"],
    visibility = ["//machina/compiler/mlir/tfrt/transforms/mlrt:__subpackages__"],
    deps = [
        ":mlrt_ops",
        ":tf_mlrt_ops",
        ":tf_mlrt_tpu_ops_inc_gen",
        "//machina/compiler/mlir/machina",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:SideEffectInterfaces",
    ],
)
