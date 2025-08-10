load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/compiler/mlir/tools/kernel_gen:friends",
        # Allow visibility from the mlir language server.
        "//learning/brain/mlir/mlir_lsp_server:__pkg__",
    ],
    licenses = ["notice"],
)

td_library(
    name = "td_files",
    srcs = [
        "tf_framework_ops.td",
        "tf_status.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@toolchain-project//mlir:AllocationOpInterfaceTdFiles",
        "@toolchain-project//mlir:ControlFlowInterfacesTdFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
        "@toolchain-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "tf_framework_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "tf_framework_ops.h.inc": ["-gen-op-decls"],
        "tf_framework_ops.cc.inc": ["-gen-op-defs"],
        "tf_framework_dialect.h.inc": ["-gen-dialect-decls"],
        "tf_framework_dialect.cc.inc": ["-gen-dialect-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "tf_framework_ops.td",
    deps = [":td_files"],
)

gentbl_cc_library(
    name = "tf_status_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "tf_status.h.inc": ["-gen-enum-decls"],
        "tf_status.cc.inc": ["-gen-enum-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "tf_status.td",
    deps = [":td_files"],
)

cc_library(
    name = "tf_framework_ops",
    srcs = [
        "tf_framework_ops.cc",
        "tf_framework_ops.cc.inc",
        "tf_framework_ops.h.inc",
    ],
    hdrs = ["tf_framework_ops.h"],
    deps = [
        ":tf_framework_ops_inc_gen",
        ":tf_status_inc_gen",
        "//machina/core/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AllocationOpInterface",
        "@toolchain-project//mlir:BufferizationDialect",
        "@toolchain-project//mlir:BytecodeOpInterface",
        "@toolchain-project//mlir:ControlFlowInterfaces",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MemRefDialect",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)
