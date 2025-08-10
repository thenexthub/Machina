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

cc_library(
    name = "atomic_rmw_utils",
    srcs = ["atomic_rmw_utils.cc"],
    hdrs = ["atomic_rmw_utils.h"],
    deps = [
        "//xla/codegen/emitters/ir:xla",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LLVMDialect",
        "@toolchain-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"passes.h.inc": [
        "-gen-pass-decls",
        "-name=Transforms",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    visibility = ["//visibility:private"],
    deps = ["@toolchain-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = [
        "erase_dead_functions.cc",
        "expand_float_ops.cc",
        "flatten_tensors.cc",
        "lower_tensors.cc",
        "lower_to_llvm.cc",
        "lower_xla_intrinsic_lib.cc",
        "lower_xla_to_scf.cc",
        "merge_pointers_to_same_slice.cc",
        "propagate_slice_indices.cc",
        "unswitch_loops.cc",
        "vectorize_loads_stores.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":atomic_rmw_utils",
        ":convert_pure_call_ops_pass",
        ":passes_inc_gen",
        ":propagate_alias_scopes",  # buildcleaner: keep
        ":simplify_affine_pass",
        ":simplify_arith_pass",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/codegen/emitters/ir:xla_cpu",
        "//xla/backends/gpu/codegen/emitters/ir:xla_gpu",
        "//xla/codegen:device_spec",
        "//xla/codegen/emitters:elemental_hlo_to_mlir",
        "//xla/codegen/emitters:implicit_arith_op_builder",
        "//xla/codegen/emitters/ir:xla",
        "//xla/codegen/intrinsic",
        "//xla/codegen/intrinsic:erf",
        "//xla/codegen/intrinsic:exp",
        "//xla/codegen/intrinsic:fptrunc",
        "//xla/codegen/intrinsic:log1p",
        "//xla/codegen/intrinsic:rsqrt",
        "//xla/codegen/intrinsic:tanh",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:map_mhlo_to_scalar_op",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:device_description_proto_cc",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AMDGPUUtils",
        "@toolchain-project//mlir:AffineToStandard",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:ArithToLLVM",
        "@toolchain-project//mlir:ArithTransforms",
        "@toolchain-project//mlir:ComplexDialect",
        "@toolchain-project//mlir:ComplexToLLVM",
        "@toolchain-project//mlir:ControlFlowToLLVM",
        "@toolchain-project//mlir:DataLayoutInterfaces",
        "@toolchain-project//mlir:DialectUtils",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncToLLVM",
        "@toolchain-project//mlir:GPUDialect",
        "@toolchain-project//mlir:GPUToNVVMTransforms",
        "@toolchain-project//mlir:GPUToROCDLTransforms",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LLVMCommonConversion",
        "@toolchain-project//mlir:LLVMDialect",
        "@toolchain-project//mlir:MathDialect",
        "@toolchain-project//mlir:MathToLLVM",
        "@toolchain-project//mlir:MathTransforms",
        "@toolchain-project//mlir:NVVMDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:ROCDLDialect",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:SCFToControlFlow",
        "@toolchain-project//mlir:SCFUtils",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:VectorDialect",
        "@toolchain-project//mlir:VectorToLLVM",
        "@toolchain-project//mlir:VectorTransforms",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "convert_pure_call_ops_pass",
    srcs = [
        "convert_pure_call_ops.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        "//xla/codegen/emitters/ir:xla",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "simplify_affine_pass",
    srcs = [
        "simplify_affine.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        "//xla/codegen/emitters/ir:xla",
        "//xla/hlo/analysis:indexing_analysis",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AffineDialect",
        "@toolchain-project//mlir:AffineUtils",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:DataLayoutInterfaces",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "simplify_arith_pass",
    srcs = [
        "simplify_arith.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        "//xla/codegen/emitters/ir:xla",
        "//xla/hlo/analysis:indexing_analysis",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:GPUDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "propagate_alias_scopes",
    srcs = ["propagate_alias_scopes.cc"],
    deps = [
        ":passes_inc_gen",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:CallOpInterfaces",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LLVMDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
    ],
)

cc_library(
    name = "pass_pipelines",
    srcs = ["pass_pipelines.cc"],
    hdrs = ["pass_pipelines.h"],
    deps = [
        ":convert_pure_call_ops_pass",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Transforms",
    ],
)
