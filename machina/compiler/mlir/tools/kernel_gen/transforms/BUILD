load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/compiler/mlir/tools/kernel_gen:friends"],
    licenses = ["notice"],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LLVMDialect",
    ],
)

cc_library(
    name = "tf_framework_legalize_to_llvm",
    srcs = ["tf_framework_legalize_to_llvm.cc"],
    hdrs = ["rewriters.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":utils",
        "//machina/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LLVMCommonConversion",
        "@toolchain-project//mlir:LLVMDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
    ],
)

cc_library(
    name = "bufferize",
    srcs = ["bufferize.cc"],
    hdrs = ["rewriters.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:BufferizationTransforms",
        "@toolchain-project//mlir:ComplexDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MemRefDialect",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@stablehlo//:chlo_ops",
    ],
)

cc_library(
    name = "embed_tf_framework",
    srcs = ["embed_tf_framework.cc"],
    hdrs = ["rewriters.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "@toolchain-project//mlir:ControlFlowDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MemRefDialect",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "kernel_gen_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"kernel_gen_passes.h.inc": [
        "-gen-pass-decls",
        "-name=KernelGen",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = ["@toolchain-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "gpu_passes",
    srcs = [
        "gpu_kernel_to_blob_pass.cc",
        "tf_kernel_to_llvm_pass.cc",
    ],
    hdrs = ["passes.h"],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]) + if_rocm_is_configured(["-DMACHINA_USE_ROCM=1"]),
    deps = [
        ":embed_tf_framework",
        ":kernel_gen_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "//machina/core:lib",
        "//machina/core/platform:errors",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:TargetParser",
        "@toolchain-project//toolchain:TransformUtils",
        "@toolchain-project//mlir:AffineDialect",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:ArithToLLVM",
        "@toolchain-project//mlir:ArithTransforms",
        "@toolchain-project//mlir:ComplexDialect",
        "@toolchain-project//mlir:ComplexToLLVM",
        "@toolchain-project//mlir:ControlFlowDialect",
        "@toolchain-project//mlir:ControlFlowToLLVM",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncToLLVM",
        "@toolchain-project//mlir:FuncTransforms",
        "@toolchain-project//mlir:GPUDialect",
        "@toolchain-project//mlir:GPUToGPURuntimeTransforms",
        "@toolchain-project//mlir:GPUTransforms",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LLVMCommonConversion",
        "@toolchain-project//mlir:LLVMDialect",
        "@toolchain-project//mlir:LinalgDialect",
        "@toolchain-project//mlir:LinalgTransforms",
        "@toolchain-project//mlir:MathDialect",
        "@toolchain-project//mlir:MathToLLVM",
        "@toolchain-project//mlir:MathToLibm",
        "@toolchain-project//mlir:MemRefDialect",
        "@toolchain-project//mlir:MemRefToLLVM",
        "@toolchain-project//mlir:MemRefTransforms",
        "@toolchain-project//mlir:NVVMToLLVMIRTranslation",  # buildcleaner: keep
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:ROCDLToLLVMIRTranslation",  # buildcleaner: keep
        "@toolchain-project//mlir:ReconcileUnrealizedCasts",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:SCFToControlFlow",
        "@toolchain-project//mlir:SCFTransforms",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:ShapeToStandard",
        "@toolchain-project//mlir:ShapeTransforms",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TensorTransforms",
        "@toolchain-project//mlir:ToLLVMIRTranslation",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@toolchain-project//mlir:VectorDialect",
        "@toolchain-project//mlir:VectorToLLVM",
        "@toolchain-project//mlir:VectorTransforms",
        "@local_xla//xla:debug_options_flags",
        "@local_xla//xla:xla_proto_cc",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@local_xla//xla/mlir_hlo:type_conversion",
        "@local_xla//xla/service/gpu:gpu_asm_opts_util",
        "@local_xla//xla/service/gpu:target_constants",
        "@local_xla//xla/stream_executor:device_description",
    ] + if_cuda_is_configured([
        "@local_tsl//tsl/platform:cuda_root_path",
        "@local_xla//xla/stream_executor/cuda:cuda_asm_compiler",
        "@local_xla//xla/service/gpu/llvm_gpu_backend:nvptx_backend",
    ]) + if_rocm_is_configured([
        "@local_xla//xla/stream_executor/gpu:asm_compiler",
        "@local_xla//xla/service/gpu/llvm_gpu_backend:amdgpu_backend",
        "//machina/core/platform:rocm_rocdl_path",
    ]),
)

cc_library(
    name = "passes",
    srcs = [
        "broadcast_propagation_pass.cc",
        "buffer_reuse_pass.cc",
        "bufferize_pass.cc",
        "copy_cleanup_pass.cc",
        "embed_tf_framework_pass.cc",
        "func_to_jit_invocations.cc",
        "fuse_inner_parallel_loops_pass.cc",
        "merge_assuming_ops_pass.cc",
        "parallel_loops_to_sequential.cc",
        "rewrite_tf_framework_assert.cc",
        "same_shape_propagation.cc",
        "shape_simplification_pass.cc",
        "shape_to_descriptors_pass.cc",
        "machina_abi_knowledge_propagation.cc",
    ],
    hdrs = ["passes.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":bufferize",  # buildcleaner: keep
        ":embed_tf_framework",  # buildcleaner: keep
        ":kernel_gen_passes_inc_gen",
        ":tf_framework_legalize_to_llvm",  # buildcleaner: keep
        "//machina/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Analysis",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:BufferizationTransforms",
        "@toolchain-project//mlir:ControlFlowDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:GPUDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:InferTypeOpInterface",
        "@toolchain-project//mlir:LLVMDialect",
        "@toolchain-project//mlir:LinalgDialect",
        "@toolchain-project//mlir:MathDialect",
        "@toolchain-project//mlir:MemRefDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:SCFToControlFlow",
        "@toolchain-project//mlir:SCFTransforms",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:ShapeToStandard",
        "@toolchain-project//mlir:ShapeTransforms",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@local_xla//xla/mlir_hlo:transforms_passes",
        "@stablehlo//:base",
    ],
)
