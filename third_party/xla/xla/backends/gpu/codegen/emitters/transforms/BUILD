load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

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
    tbl_outs = {"passes.h.inc": [
        "-gen-pass-decls",
        "-name=GpuFusionTransforms",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    visibility = ["//visibility:private"],
    deps = ["@toolchain-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = [
        "convert_float_amd.cc",
        "convert_float_nvidia.cc",
        "convert_index_type.cc",
        "lower_xla_shared.cc",
        "optimize_loops.cc",
        "peel_loops.cc",
        "recover_exp2.cc",
    ],
    hdrs = ["passes.h"],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]) + if_rocm_is_configured(["-DMACHINA_USE_ROCM=1"]),
    deps = [
        ":passes_inc_gen",
        "//xla:shape_util",
        "//xla:util",
        "//xla/backends/gpu/codegen/emitters/ir:xla_gpu",
        "//xla/codegen/emitters/ir:xla",
        "//xla/codegen/emitters/ir:xla_dialect_inc_gen",
        "//xla/codegen/emitters/transforms:atomic_rmw_utils",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:map_mhlo_to_scalar_op",
        "//xla/service/gpu:gpu_fusible",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:semantic_version",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AffineDialect",
        "@toolchain-project//mlir:AffineUtils",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:CallOpInterfaces",
        "@toolchain-project//mlir:ComplexDialect",
        "@toolchain-project//mlir:DataLayoutInterfaces",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:GPUDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LLVMDialect",
        "@toolchain-project//mlir:MathDialect",
        "@toolchain-project//mlir:MathTransforms",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:SCFUtils",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:VectorDialect",
        "@local_tsl//tsl/platform:protobuf",
    ] + if_cuda_is_configured([
        "//xla/service/gpu/llvm_gpu_backend:nvptx_backend",
    ]) + if_rocm_is_configured([
        "//xla/service/gpu/llvm_gpu_backend:amdgpu_backend",
    ]),
)
