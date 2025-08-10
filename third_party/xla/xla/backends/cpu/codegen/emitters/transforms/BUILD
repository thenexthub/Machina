load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library")
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

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes.h.inc": [
        "-gen-pass-decls",
        "-name=XlaCpuTransforms",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    visibility = ["//visibility:private"],
    deps = ["@toolchain-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = [
        "add_loop_unroll_flags.cc",
        "add_reduction_fast_math_flags.cc",
        "expand_float_ops.cc",
        "lower_to_llvm.cc",
        "lower_xla_shared.cc",
        "peel_workgroup_loop.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        ":xla_cpu_rewrite_patterns",
        "//xla/backends/cpu/codegen/emitters/ir:xla_cpu",
        "//xla/codegen/emitters:implicit_arith_op_builder",
        "//xla/codegen/emitters/ir:xla",
        "//xla/codegen/intrinsic:fptrunc",
        "//xla/codegen/intrinsic:log1p",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/mlir/utils:type_util",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:DataLayoutInterfaces",
        "@toolchain-project//mlir:DialectUtils",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LLVMDialect",
        "@toolchain-project//mlir:MathDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:VectorDialect",
    ],
)

cc_library(
    name = "xla_cpu_rewrite_patterns",
    srcs = ["xla_cpu_rewrite_patterns.cc"],
    hdrs = ["xla_cpu_rewrite_patterns.h"],
    deps = [
        "//xla/backends/cpu/codegen/emitters/ir:xla_cpu",
        "//xla/codegen/emitters/ir:xla",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:DataLayoutInterfaces",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LLVMDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)
