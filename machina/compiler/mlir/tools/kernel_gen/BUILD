load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load(
    "@local_xla//xla/stream_executor:build_defs.bzl",
    "if_gpu_is_configured",
)
load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "check_deps",
    "tf_cc_binary",
)
load(
    "//machina/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//machina/core/platform:build_config_root.bzl",
    "if_llvm_aarch32_available",
    "if_llvm_aarch64_available",
    "if_llvm_powerpc_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # Edge TPU compiler needs to use some compiler passes from kernel_gen.
        "//platforms/darwinn/compiler/...",
        "//machina/compiler/...",
        "//machina/core/kernels/mlir_generated/...",
    ],
)

cc_library(
    name = "kernel_creator",
    srcs = ["kernel_creator.cc"],
    hdrs = ["kernel_creator.h"],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]) + if_rocm_is_configured(["-DMACHINA_USE_ROCM=1"]),
    deps = [
        "//machina/compiler/mlir/machina:dump_mlir_util",
        "//machina/compiler/mlir/tools/kernel_gen/transforms:bufferize",
        "//machina/compiler/mlir/tools/kernel_gen/transforms:gpu_passes",  # fixdeps: keep
        "//machina/compiler/mlir/tools/kernel_gen/transforms:passes",
        "//machina/core:lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AffineToStandard",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:ArithTransforms",
        "@toolchain-project//mlir:BufferizationTransforms",
        "@toolchain-project//mlir:BuiltinToLLVMIRTranslation",
        "@toolchain-project//mlir:ComplexDialect",
        "@toolchain-project//mlir:ComplexToStandard",
        "@toolchain-project//mlir:ControlFlowDialect",
        "@toolchain-project//mlir:DataLayoutInterfaces",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:GPUDialect",
        "@toolchain-project//mlir:GPUToGPURuntimeTransforms",
        "@toolchain-project//mlir:GPUToLLVMIRTranslation",
        "@toolchain-project//mlir:GPUToNVVMTransforms",
        "@toolchain-project//mlir:GPUTransforms",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LLVMIRTransforms",
        "@toolchain-project//mlir:LLVMToLLVMIRTranslation",
        "@toolchain-project//mlir:LinalgTransforms",
        "@toolchain-project//mlir:MemRefDialect",
        "@toolchain-project//mlir:MemRefTransforms",
        "@toolchain-project//mlir:NVVMToLLVMIRTranslation",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:ROCDLToLLVMIRTranslation",
        "@toolchain-project//mlir:ReconcileUnrealizedCasts",
        "@toolchain-project//mlir:SCFToControlFlow",
        "@toolchain-project//mlir:SCFToGPU",
        "@toolchain-project//mlir:SCFTransforms",
        "@toolchain-project//mlir:ShapeDialect",
        "@toolchain-project//mlir:ShapeToStandard",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@toolchain-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo",
        "@local_xla//xla/mlir_hlo:all_passes",  # fixdeps: keep
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@local_xla//xla/mlir_hlo:transforms_gpu_passes",
        "@local_xla//xla/mlir_hlo:transforms_passes",
        "@stablehlo//:chlo_ops",
    ],
)

check_deps(
    name = "hlo_to_kernel_check_deps",
    disallowed_deps = ["@local_xla//xla:literal"],
    deps = [":hlo_to_kernel"],
)

tf_cc_binary(
    name = "hlo_to_kernel",
    srcs = ["hlo_to_kernel.cc"],
    visibility = [
        "//machina/compiler/mlir/tools/kernel_gen/tests/hlo_to_kernel:__pkg__",
        "//machina/core/kernels/mlir_generated:__pkg__",
    ],
    deps = [
        ":kernel_creator",
        "//machina/compiler/mlir:init_mlir",
        "//machina/compiler/mlir/machina",
        "//machina/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Analysis",
        "@toolchain-project//toolchain:CodeGen",
        "@toolchain-project//toolchain:Core",
        "@toolchain-project//toolchain:MC",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:Target",
        "@toolchain-project//toolchain:TargetParser",
        "@toolchain-project//mlir:BufferizationInterfaces",
        "@toolchain-project//mlir:ExecutionEngineUtils",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LLVMToLLVMIRTranslation",
        "@toolchain-project//mlir:MemRefTransforms",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:ToLLVMIRTranslation",
        "@local_xla//xla/service/llvm_ir:llvm_command_line_options",
    ] + if_llvm_aarch32_available([
        "@toolchain-project//toolchain:ARMCodeGen",  # fixdeps: keep
    ]) + if_llvm_aarch64_available([
        "@toolchain-project//toolchain:AArch64CodeGen",  # fixdeps: keep
    ]) + if_llvm_powerpc_available([
        "@toolchain-project//toolchain:PowerPCCodeGen",  # fixdeps: keep
    ]) + if_llvm_system_z_available([
        "@toolchain-project//toolchain:SystemZCodeGen",  # fixdeps: keep
    ]) + if_llvm_x86_available([
        "@toolchain-project//toolchain:X86CodeGen",  # fixdeps: keep
        "@toolchain-project//toolchain:X86Disassembler",  # fixdeps: keep
    ]),
)

tf_cc_binary(
    name = "kernel-gen-opt",
    srcs = ["tools/kernel-gen-opt/kernel-gen-opt.cc"],
    visibility = ["//machina/compiler/mlir/tools/kernel_gen/tests:__subpackages__"],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "//machina/compiler/mlir/tools/kernel_gen/transforms:gpu_passes",  # fixdeps: keep
        "//machina/compiler/mlir/tools/kernel_gen/transforms:passes",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AllPassesAndDialects",
        "@toolchain-project//mlir:MlirOptLib",
        "@toolchain-project//mlir:RegisterAllDialects",
        "@toolchain-project//mlir:RegisterAllPasses",
        "@local_xla//xla/mlir_hlo:all_passes",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@stablehlo//:register",
    ],
)

exports_files(["tf_framework_c_interface.h"])

cc_library(
    name = "tf_framework_c_interface",
    srcs = ["tf_framework_c_interface.cc"],
    hdrs = ["tf_framework_c_interface.h"],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]) + if_rocm_is_configured(["-DMACHINA_USE_ROCM=1"]),
    deps = [
        ":compile_cache_item_proto_cc",
        ":kernel_creator",
        ":tf_gpu_runtime_wrappers",
        ":tf_jit_cache",
        "//machina/compiler/mlir/tools/kernel_gen/ir:tf_framework_ops",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/platform:refcount",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:JITLink",
        "@toolchain-project//toolchain:OrcShared",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ExecutionEngine",
        "@toolchain-project//mlir:ExecutionEngineUtils",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:MemRefTransforms",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:mlir_c_runner_utils",
        "@toolchain-project//mlir:mlir_runner_utils",
        "@local_xla//xla/stream_executor:stream",
    ],
)

cc_library(
    name = "tf_jit_cache",
    srcs = ["tf_jit_cache.cc"],
    hdrs = ["tf_jit_cache.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:framework_lite",
        "//machina/core/framework:resource_base",
        "//machina/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ExecutionEngine",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "tf_gpu_runtime_wrappers",
    srcs = if_gpu_is_configured([
        "tf_gpu_runtime_wrappers.cc",
    ]),
    hdrs =
        if_gpu_is_configured([
            "tf_gpu_runtime_wrappers.h",
        ]),
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]) + if_rocm_is_configured([
        "-DMACHINA_USE_ROCM=1",
    ]),
    deps = [
        "//machina/core:framework",
        "//machina/core/framework:resource_base",
        "//machina/core/platform:logging",
        "//machina/core/platform:mutex",
        "//machina/core/platform:refcount",
        "//machina/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//mlir:mlir_runner_utils",
        "@local_tsl//tsl/platform:hash",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/stream_executor:stream",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_xla//xla/stream_executor/cuda:stream_executor_cuda",
    ]) + if_rocm_is_configured([
        "@local_config_rocm//rocm:rocm_headers",
        "@local_xla//xla/stream_executor/rocm:stream_executor_rocm",
    ]),
)

tf_proto_library(
    name = "compile_cache_item_proto",
    srcs = ["compile_cache_item.proto"],
)
