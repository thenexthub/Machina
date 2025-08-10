load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
    name = "xla_td_files",
    srcs = glob(["*.td"]),
    compatible_with = get_compatible_with_portable(),
    includes = ["."],
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
    name = "xla_dialect_inc_gen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    tbl_outs = {
        "xla_dialect.h.inc": ["-gen-dialect-decls"],
        "xla_dialect.cc.inc": ["-gen-dialect-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "xla_dialect.td",
    deps = [":xla_td_files"],
)

gentbl_cc_library(
    name = "xla_ops_inc_gen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    tbl_outs = {
        "xla_ops.h.inc": ["-gen-op-decls"],
        "xla_ops.cc.inc": ["-gen-op-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "xla_ops.td",
    deps = [":xla_td_files"],
)

gentbl_cc_library(
    name = "xla_attrs_inc_gen",
    compatible_with = get_compatible_with_portable(),
    strip_include_prefix = ".",
    tbl_outs = {
        "xla_enums.h.inc": ["-gen-enum-decls"],
        "xla_enums.cc.inc": ["-gen-enum-defs"],
        "xla_attrs.h.inc": [
            "-gen-attrdef-decls",
        ],
        "xla_attrs.cc.inc": [
            "-gen-attrdef-defs",
        ],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "xla_attrs.td",
    deps = [":xla_td_files"],
)

cc_library(
    name = "xla",
    srcs = [
        "xla_attrs.cc",
        "xla_dialect.cc",
        "xla_ops.cc",
    ],
    hdrs = [
        "xla_dialect.h",
        "xla_ops.h",
    ],
    deps = [
        ":xla_attrs_inc_gen",
        ":xla_dialect_inc_gen",
        ":xla_ops_inc_gen",
        "//xla/codegen/emitters:type_util",
        "//xla/hlo/analysis:indexing_analysis",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:BytecodeOpInterface",
        "@toolchain-project//mlir:CallOpInterfaces",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:InliningUtils",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
    ],
)

xla_test(
    name = "xla_ops_test",
    srcs = ["xla_ops_test.cc"],
    backends = ["cpu"],
    tags = ["test_migrated_to_hlo_runner_pjrt"],
    deps = [
        ":xla",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/hlo/testlib:filecheck",
        "//xla/mlir/utils:error_util",
        "//xla/tests:hlo_pjrt_test_base",
        "//xla/tests:xla_internal_test_main",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
    ],
)
