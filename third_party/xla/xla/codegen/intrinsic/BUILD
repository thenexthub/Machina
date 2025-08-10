load("//xla:xla.default.bzl", "xla_cc_test")
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
    name = "intrinsic",
    srcs = ["intrinsic.cc"],
    hdrs = ["intrinsic.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/mlir/utils:type_util",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:Target",
        "@toolchain-project//toolchain:ir_headers",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "intrinsic_test",
    srcs = ["intrinsic_test.cc"],
    deps = [
        ":intrinsic",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fptrunc",
    srcs = ["fptrunc.cc"],
    hdrs = ["fptrunc.h"],
    deps = [
        ":intrinsic",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

xla_cc_test(
    name = "fptrunc_test",
    srcs = ["fptrunc_test.cc"],
    deps = [
        ":fptrunc",
        ":intrinsic",
        "//xla:xla_data_proto_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ldexp",
    srcs = ["ldexp.cc"],
    hdrs = ["ldexp.h"],
    deps = [
        ":intrinsic",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Core",  # buildcleaner: keep
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

xla_cc_test(
    name = "ldexp_test",
    srcs = ["ldexp_test.cc"],
    deps = [
        ":intrinsic",
        ":ldexp",
        ":simple_jit_runner",
        ":test_matchers",
        "//xla:xla_data_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:JITLink",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

cc_library(
    name = "simple_jit_runner",
    testonly = True,
    srcs = ["simple_jit_runner.cc"],
    hdrs = ["simple_jit_runner.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu/codegen:contiguous_section_memory_manager",
        "//xla/service/cpu:orc_jit_memory_mapper",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@toolchain-project//toolchain:ExecutionEngine",
        "@toolchain-project//toolchain:JITLink",
        "@toolchain-project//toolchain:MC",
        "@toolchain-project//toolchain:OrcJIT",  # buildcleaner: keep
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:Target",
        "@toolchain-project//toolchain:TargetParser",
        "@toolchain-project//toolchain:ir_headers",
    ] + if_llvm_aarch64_available([
        "@toolchain-project//toolchain:AArch64AsmParser",  # fixdeps: keep
        "@toolchain-project//toolchain:AArch64CodeGen",  # fixdeps: keep
    ]) + if_llvm_powerpc_available([
        "@toolchain-project//toolchain:PowerPCAsmParser",  # fixdeps: keep
        "@toolchain-project//toolchain:PowerPCCodeGen",  # fixdeps: keep
    ]) + if_llvm_system_z_available([
        "@toolchain-project//toolchain:SystemZAsmParser",  # fixdeps: keep
        "@toolchain-project//toolchain:SystemZCodeGen",  # fixdeps: keep
    ]) + if_llvm_x86_available([
        "@toolchain-project//toolchain:X86AsmParser",  # fixdeps: keep
        "@toolchain-project//toolchain:X86CodeGen",  # fixdeps: keep
    ]),
)

xla_cc_test(
    name = "simple_jit_runner_test",
    srcs = ["simple_jit_runner_test.cc"],
    deps = [
        ":simple_jit_runner",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

cc_library(
    name = "test_matchers",
    testonly = True,
    srcs = [],
    hdrs = ["test_matchers.h"],
    deps = [
        "//xla:fp_util",
        "@com_google_googletest//:gtest_for_library",
    ],
)

cc_library(
    name = "vec_name_mangler",
    hdrs = ["vec_name_mangler.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "vec_name_mangler_test",
    srcs = ["vec_name_mangler_test.cc"],
    deps = [
        ":vec_name_mangler",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "string_interner",
    hdrs = ["string_interner.h"],
    deps = ["@com_google_absl//absl/strings:string_view"],
)

cc_library(
    name = "intrinsic_compiler_lib",
    srcs = ["intrinsic_compiler_lib.cc"],
    hdrs = ["intrinsic_compiler_lib.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Analysis",
        "@toolchain-project//toolchain:IPO",
        "@toolchain-project//toolchain:InstCombine",
        "@toolchain-project//toolchain:Passes",
        "@toolchain-project//toolchain:Scalar",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

xla_cc_test(
    name = "intrinsic_compiler_lib_test",
    srcs = ["intrinsic_compiler_lib_test.cc"],
    deps = [
        ":intrinsic_compiler_lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

cc_library(
    name = "exp",
    srcs = ["exp.cc"],
    hdrs = ["exp.h"],
    deps = [
        ":intrinsic",
        ":ldexp",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Analysis",
        "@toolchain-project//toolchain:Core",  # buildcleaner: keep
        "@toolchain-project//toolchain:ExecutionEngine",
        "@toolchain-project//toolchain:IPO",
        "@toolchain-project//toolchain:JITLink",
        "@toolchain-project//toolchain:Passes",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:Target",
        "@toolchain-project//toolchain:TransformUtils",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

cc_library(
    name = "log1p",
    srcs = ["log1p.cc"],
    hdrs = ["log1p.h"],
    deps = [
        ":intrinsic",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

xla_cc_test(
    name = "log1p_test",
    srcs = ["log1p_test.cc"],
    deps = [
        ":intrinsic",
        ":log1p",
        ":simple_jit_runner",
        ":test_matchers",
        "//xla:xla_data_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:JITLink",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

xla_cc_test(
    name = "exp_test",
    srcs = ["exp_test.cc"],
    deps = [
        ":exp",
        ":intrinsic",
        ":ldexp",
        ":simple_jit_runner",
        ":test_matchers",
        "//xla:xla_data_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:JITLink",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

cc_library(
    name = "rsqrt",
    srcs = ["rsqrt.cc"],
    hdrs = ["rsqrt.h"],
    deps = [
        ":intrinsic",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Core",  # buildcleaner: keep
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:Target",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

cc_library(
    name = "erf",
    srcs = ["erf.cc"],
    hdrs = ["erf.h"],
    deps = [
        ":intrinsic",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

xla_cc_test(
    name = "erf_test",
    srcs = ["erf_test.cc"],
    deps = [
        ":erf",
        ":intrinsic",
        ":simple_jit_runner",
        ":test_matchers",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:JITLink",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

xla_cc_test(
    name = "rsqrt_test",
    srcs = ["rsqrt_test.cc"],
    deps = [
        ":intrinsic",
        ":rsqrt",
        ":simple_jit_runner",
        ":test_matchers",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:JITLink",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:TargetParser",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

xla_cc_test(
    name = "rsqrt_benchmark_test",
    srcs = ["rsqrt_benchmark_test.cc"],
    fail_if_no_test_linked = False,  # NOLINT=This contains benchmarks only, no tests.
    fail_if_no_test_selected = False,  # NOLINT=This contains benchmarks only, no tests.
    deps = [
        ":intrinsic",
        ":rsqrt",
        ":simple_jit_runner",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@toolchain-project//toolchain:Target",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

cc_library(
    name = "tanh",
    srcs = ["tanh.cc"],
    hdrs = ["tanh.h"],
    deps = [
        ":intrinsic",
        "//xla:xla_data_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:ir_headers",
    ],
)

xla_cc_test(
    name = "tanh_test",
    srcs = ["tanh_test.cc"],
    deps = [
        ":intrinsic",
        ":simple_jit_runner",
        ":tanh",
        ":test_matchers",
        "//xla:xla_data_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:JITLink",
        "@toolchain-project//toolchain:ir_headers",
    ],
)
