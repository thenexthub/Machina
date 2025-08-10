load("//xla:py_strict.bzl", "py_strict_test")
load("//xla:xla.default.bzl", "xla_cc_test", "xla_internal")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "if_llvm_aarch64_available",
    "if_llvm_powerpc_available",
    "if_llvm_system_z_available",
    "if_llvm_x86_available",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = internal_visibility(["//xla/backends/cpu:xla_backend_cpu_internal_access"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "contiguous_section_memory_manager",
    srcs = ["contiguous_section_memory_manager.cc"],
    hdrs = ["contiguous_section_memory_manager.h"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@toolchain-project//toolchain:ExecutionEngine",
        "@toolchain-project//toolchain:Support",
        # TODO(basioli): This dependency increases the binary size significantly.
        # Consider reducing the dependency size, or use something alternative.
        "//xla:util",
    ],
)

cc_library(
    name = "cpu_features",
    srcs = ["cpu_features.cc"],
    hdrs = ["cpu_features.h"],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:TargetParser",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "ir_compiler",
    srcs = ["ir_compiler.cc"],
    hdrs = ["ir_compiler.h"],
    deps = [
        ":cpu_features",
        ":kernel_api_ir_builder",
        ":polynomial_approximations",
        "//xla:util",
        "//xla:xla_proto_cc",
        "//xla/codegen:intrinsic_lib",
        "//xla/codegen/intrinsic:intrinsic_compiler_lib",
        "//xla/service:hlo_module_config",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/service/cpu:cpu_options",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@toolchain-project//toolchain:Analysis",
        "@toolchain-project//toolchain:Core",
        "@toolchain-project//toolchain:ExecutionEngine",
        "@toolchain-project//toolchain:IPO",
        "@toolchain-project//toolchain:Instrumentation",
        "@toolchain-project//toolchain:MC",
        "@toolchain-project//toolchain:Object",
        "@toolchain-project//toolchain:OrcJIT",
        "@toolchain-project//toolchain:Passes",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:Target",
        "@toolchain-project//toolchain:TargetParser",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

xla_cc_test(
    name = "ir_compiler_test",
    srcs = ["ir_compiler_test.cc"],
    deps = [
        ":ir_compiler",
        ":kernel_api_ir_builder",
        "//xla:util",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:AsmParser",
        "@toolchain-project//toolchain:Core",
        "@toolchain-project//toolchain:MC",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:Target",
        "@toolchain-project//toolchain:TargetParser",
    ] + if_llvm_aarch64_available([
        "@toolchain-project//toolchain:AArch64CodeGen",  # fixdeps: keep
    ]) + if_llvm_powerpc_available([
        "@toolchain-project//toolchain:PowerPCCodeGen",  # fixdeps: keep
    ]) + if_llvm_system_z_available([
        "@toolchain-project//toolchain:SystemZCodeGen",  # fixdeps: keep
    ]) + if_llvm_x86_available([
        "@toolchain-project//toolchain:X86CodeGen",  # fixdeps: keep
    ]),
)

cc_library(
    name = "fusion_compiler",
    srcs = ["fusion_compiler.cc"],
    hdrs = ["fusion_compiler.h"],
    deps = [
        ":kernel_api_ir_builder",
        "//xla:status_macros",
        "//xla:util",
        "//xla/backends/cpu/codegen/emitters/ir:xla_cpu",
        "//xla/backends/cpu/codegen/emitters/transforms:passes",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/codegen:mlir_kernel_source",
        "//xla/codegen:trace_pass_instrumentation",
        "//xla/codegen/emitters/ir:xla",
        "//xla/codegen/emitters/ir:xla_attrs_inc_gen",
        "//xla/codegen/emitters/transforms:pass_pipelines",
        "//xla/codegen/emitters/transforms:passes",
        "//xla/mlir/tools/mlir_replay/public:compiler_trace_proto_cc",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/tsl/framework/mlir:status_scoped_diagnostic_handler",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Core",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:AffineDialect",
        "@toolchain-project//mlir:AffineToStandard",
        "@toolchain-project//mlir:AffineTransforms",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:BuiltinToLLVMIRTranslation",
        "@toolchain-project//mlir:ComplexToStandard",
        "@toolchain-project//mlir:ControlFlowDialect",
        "@toolchain-project//mlir:DLTIDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LLVMDialect",
        "@toolchain-project//mlir:LLVMIRTransforms",
        "@toolchain-project//mlir:LLVMToLLVMIRTranslation",
        "@toolchain-project//mlir:MathDialect",
        "@toolchain-project//mlir:MathToLLVM",
        "@toolchain-project//mlir:MemRefTransforms",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:ReconcileUnrealizedCasts",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:SCFToControlFlow",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:ToLLVMIRTranslation",
        "@toolchain-project//mlir:Transforms",
        "@toolchain-project//mlir:VectorDialect",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
    ],
)

cc_library(
    name = "jit_compiler",
    srcs = ["jit_compiler.cc"],
    hdrs = ["jit_compiler.h"],
    deps = [
        ":compiled_function_library",
        ":contiguous_section_memory_manager",
        ":cpu_features",
        ":execution_engine",
        ":ir_compiler",
        ":object_loader",
        "//xla:util",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/service/cpu:orc_jit_memory_mapper",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Core",
        "@toolchain-project//toolchain:ExecutionEngine",
        "@toolchain-project//toolchain:MC",
        "@toolchain-project//toolchain:OrcJIT",
        "@toolchain-project//toolchain:OrcShared",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:Target",
        "@toolchain-project//toolchain:TargetParser",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
    ] + if_llvm_aarch64_available([
        "@toolchain-project//toolchain:AArch64CodeGen",  # fixdeps: keep
    ]) + if_llvm_powerpc_available([
        "@toolchain-project//toolchain:PowerPCCodeGen",  # fixdeps: keep
    ]) + if_llvm_system_z_available([
        "@toolchain-project//toolchain:SystemZCodeGen",  # fixdeps: keep
    ]) + if_llvm_x86_available([
        "@toolchain-project//toolchain:X86CodeGen",  # fixdeps: keep
    ]) + xla_internal(["service/cpu:named_orc_jit_memory_mapper"]),
)

xla_cc_test(
    name = "jit_compiler_test",
    srcs = ["jit_compiler_test.cc"],
    deps = [
        ":ir_compiler",
        ":jit_compiler",
        ":kernel_api_ir_builder",
        "//xla:util",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:AsmParser",
        "@toolchain-project//toolchain:Core",
        "@toolchain-project//toolchain:OrcJIT",
        "@toolchain-project//toolchain:OrcShared",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:Target",
    ],
)

cc_library(
    name = "polynomial_approximations",
    srcs = ["polynomial_approximations.cc"],
    hdrs = ["polynomial_approximations.h"],
    deps = [
        ":vector_ir_builder",
        "//xla:xla_data_proto_cc",
        "//xla/codegen/intrinsic:string_interner",
        "//xla/codegen/intrinsic:vec_name_mangler",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Analysis",
        "@toolchain-project//toolchain:Core",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:TransformUtils",
    ],
)

cc_library(
    name = "target_machine_features",
    srcs = ["target_machine_features.cc"],
    hdrs = ["target_machine_features.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu:alignment",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Analysis",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:Target",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

cc_library(
    name = "target_machine_test_base",
    testonly = True,
    hdrs = ["target_machine_test_base.h"],
    deps = [
        ":target_machine_features",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:ARMCodeGen",  # fixdeps: keep
        "@toolchain-project//toolchain:ExecutionEngine",
        "@toolchain-project//toolchain:MC",
        "@toolchain-project//toolchain:Target",
        "@toolchain-project//toolchain:TargetParser",
        "@toolchain-project//toolchain:X86CodeGen",  # fixdeps: keep
    ],
)

xla_cc_test(
    name = "target_machine_features_test",
    srcs = ["target_machine_features_test.cc"],
    deps = [
        ":target_machine_features",
        ":target_machine_test_base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "vector_ir_builder",
    srcs = ["vector_ir_builder.cc"],
    hdrs = ["vector_ir_builder.h"],
    deps = [
        ":target_machine_features",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Core",
        "@toolchain-project//toolchain:Support",
    ],
)

cc_library(
    name = "kernel_api_ir_builder",
    srcs = ["kernel_api_ir_builder.cc"],
    hdrs = ["kernel_api_ir_builder.h"],
    deps = [
        ":symbol_name_util",
        "//xla:cpu_function_runtime",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:buffer_assignment",
        "//xla/service:hlo_module_config",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

xla_cc_test(
    name = "kernel_api_ir_builder_test",
    srcs = ["kernel_api_ir_builder_test.cc"],
    deps = [
        ":kernel_api_ir_builder",
        "//xla:cpu_function_runtime",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/analysis:alias_info",
        "//xla/hlo/analysis:hlo_ordering",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:filecheck",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/service:buffer_assignment",
        "//xla/service:buffer_value",
        "//xla/service:hlo_module_config",
        "//xla/service:logical_buffer",
        "//xla/service/cpu:cpu_executable",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

cc_library(
    name = "compiled_function_library",
    srcs = ["compiled_function_library.cc"],
    hdrs = ["compiled_function_library.h"],
    deps = [
        ":execution_engine",
        "//xla/backends/cpu/runtime:function_library",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

xla_cc_test(
    name = "aot_compiled_function_library_test",
    srcs = ["aot_compiled_function_library_test.cc"],
    deps = [
        ":aot_compiled_function_library",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "aot_compiled_function_library",
    srcs = ["aot_compiled_function_library.cc"],
    hdrs = ["aot_compiled_function_library.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/backends/cpu/runtime:function_library",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "object_loader",
    srcs = ["object_loader.cc"],
    hdrs = ["object_loader.h"],
    deps = [
        ":compiled_function_library",
        ":contiguous_section_memory_manager",
        ":execution_engine",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/service/cpu:orc_jit_memory_mapper",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:JITLink",
        "@toolchain-project//toolchain:OrcShared",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

cc_library(
    name = "execution_engine",
    srcs = ["execution_engine.cc"],
    hdrs = ["execution_engine.h"],
    deps = [
        ":contiguous_section_memory_manager",
        "//xla/service/cpu:orc_jit_memory_mapper",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@toolchain-project//toolchain:ExecutionEngine",
        "@toolchain-project//toolchain:JITLink",
        "@toolchain-project//toolchain:OrcJIT",  # buildcleaner: keep
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

cc_library(
    name = "computation_kernel_emitter",
    srcs = ["computation_kernel_emitter.cc"],
    hdrs = ["computation_kernel_emitter.h"],
    deps = [
        ":kernel_api_ir_builder",
        ":target_machine_features",
        "//xla:shape_util",
        "//xla:util",
        "//xla/codegen:kernel_definition",
        "//xla/codegen:kernel_emitter",
        "//xla/codegen:kernel_spec",
        "//xla/codegen:llvm_ir_kernel_source",
        "//xla/codegen:llvm_kernel_definition",
        "//xla/codegen:llvm_kernel_emitter",
        "//xla/hlo/ir:hlo",
        "//xla/runtime:work_group",
        "//xla/service:buffer_assignment",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/service/cpu:ir_emitter",
        "//xla/service/llvm_ir:ir_array",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

py_strict_test(
    name = "scatter_kernel_emitter_test",
    srcs = ["scatter_kernel_emitter_test.py"],
    tags = [
        "no_oss",
    ],
    deps = [
        "//third_party/py/numpy",
        "//xla/backends/cpu/testlib",
        "//xla/codegen/testlib",
        "//xla/python:xla_extension",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

xla_cc_test(
    name = "object_loader_test",
    srcs = ["object_loader_test.cc"],
    deps = [
        ":execution_engine",
        ":ir_compiler",
        ":jit_compiler",
        ":object_loader",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/codegen:kernel_api_ir_builder",
        "//xla/backends/cpu/runtime:function_library",
        "//xla/service:cpu_plugin",
        "//xla/service/cpu:executable_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:AsmParser",
        "@toolchain-project//toolchain:JITLink",
        "@toolchain-project//toolchain:Object",
        "@toolchain-project//toolchain:OrcShared",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:Target",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

cc_library(
    name = "symbol_name_util",
    srcs = ["symbol_name_util.cc"],
    hdrs = ["symbol_name_util.h"],
    deps = [
        "//xla:util",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fusion_emitter",
    srcs = ["fusion_emitter.cc"],
    hdrs = ["fusion_emitter.h"],
    deps = [
        ":kernel_api_ir_builder",
        ":symbol_name_util",
        "//xla:shape_util",
        "//xla:util",
        "//xla/backends/cpu:alignment",
        "//xla/codegen:hlo_fusion_spec",
        "//xla/codegen:mlir_kernel_definition",
        "//xla/codegen/emitters:concatenate_kernel_emitter",
        "//xla/codegen/emitters:kernel_arguments",
        "//xla/codegen/emitters:loop_kernel_emitter",
        "//xla/codegen/emitters/ir:xla",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_traversal",
        "//xla/runtime:work_cluster",
        "//xla/runtime:work_dimensions",
        "//xla/runtime:work_group",
        "//xla/runtime:work_item",
        "//xla/runtime:work_tile_size",
        "//xla/service:buffer_assignment",
        "//xla/service/cpu:backend_config_proto_cc",
        "//xla/service/gpu:ir_emission_utils",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
    ],
)

xla_cc_test(
    name = "symbol_name_util_test",
    srcs = ["symbol_name_util_test.cc"],
    deps = [
        ":symbol_name_util",
        "//xla/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

py_strict_test(
    name = "computation_kernel_emitter_test",
    srcs = ["computation_kernel_emitter_test.py"],
    tags = [
        "no_oss",
    ],
    deps = [
        "//third_party/py/numpy",
        "//xla/backends/cpu/testlib",
        "//xla/codegen/testlib",
        "//xla/python:xla_extension",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "fusion_emitter_test",
    srcs = ["fusion_emitter_test.py"],
    tags = [
        "no_oss",
    ],
    deps = [
        "//third_party/py/numpy",
        "//xla/backends/cpu/testlib",
        "//xla/codegen/testlib",
        "//xla/python:xla_extension",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
