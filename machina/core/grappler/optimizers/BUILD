load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("@local_xla//xla/tsl/mkl:build_defs.bzl", "if_mkl")
load("//machina:machina.bzl", "tf_cc_test", "tf_cc_test_mkl", "tf_copts", "tf_cuda_cc_test")
load("//machina:machina.default.bzl", "filegroup", "tf_kernel_library")

# Platform specific build config
load(
    "//machina/core/platform:build_config_root.bzl",
    "if_static",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "graph_optimizer.h",
        "meta_optimizer.h",
    ],
    visibility = [
        "//machina/python/grappler:__pkg__",
    ],
)

cc_library(
    name = "static_schedule",
    srcs = ["static_schedule.cc"],
    hdrs = [
        "static_schedule.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/clusters:cluster",
        "//machina/core/grappler/costs:cost_estimator",
        "//machina/core/grappler/costs:graph_properties",
        "//machina/core/grappler/costs:op_level_cost_estimator",
        "//machina/core/grappler/costs:virtual_placer",
    ],
)

tf_cuda_cc_test(
    name = "static_schedule_test",
    srcs = ["static_schedule_test.cc"],
    deps = [
        ":static_schedule",
        "//machina/cc:cc_ops",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/clusters:virtual_cluster",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "auto_parallel",
    srcs = ["auto_parallel.cc"],
    hdrs = [
        "auto_parallel.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:devices",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/clusters:cluster",
        "//machina/core/grappler/utils:transitive_fanin",
    ],
)

tf_cuda_cc_test(
    name = "auto_parallel_test",
    srcs = ["auto_parallel_test.cc"],
    deps = [
        ":auto_parallel",
        "//machina/cc:cc_ops",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
    ],
)

cc_library(
    name = "constant_folding",
    srcs = ["constant_folding.cc"],
    hdrs = [
        "constant_folding.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":evaluation_utils",
        ":graph_optimizer",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/clusters:cluster",
        "//machina/core/grappler/costs:graph_properties",
        "//machina/core/grappler/utils:symbolic_shapes",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "constant_folding_test",
    srcs = ["constant_folding_test.cc"],
    shard_count = 5,
    # Running cuda on cpu will trigger tests guarded by GOOGLE_CUDA but NCHW
    # won't be available, which result in test failures. So disable that.
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":constant_folding",
        ":dependency_optimizer",
        "//machina/cc:array_ops_internal",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:direct_session",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "function_optimizer",
    srcs = ["function_optimizer.cc"],
    hdrs = [
        "function_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//machina/compiler/jit:common",
        "//machina/core:core_cpu_base",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:graph_view",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/utils:functions",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "function_optimizer_test",
    srcs = ["function_optimizer_test.cc"],
    deps = [
        ":function_optimizer",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/cc:functional_ops",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:direct_session",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/utils:grappler_test",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "graph_optimizer",
    hdrs = [
        "graph_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "graph_optimizer_stage",
    srcs = ["graph_optimizer_stage.cc"],
    hdrs = ["graph_optimizer_stage.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/costs:graph_properties",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "graph_optimizer_stage_test",
    size = "small",
    srcs = ["graph_optimizer_stage_test.cc"],
    deps = [
        ":graph_optimizer_stage",
        "//machina/cc:cc_ops",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/costs:graph_properties",
        "//machina/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "custom_graph_optimizer",
    hdrs = [
        "custom_graph_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ],
)

cc_library(
    name = "arithmetic_optimizer",
    srcs = ["arithmetic_optimizer.cc"],
    hdrs = [
        "arithmetic_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":constant_folding",
        ":graph_optimizer",
        ":graph_optimizer_stage",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:graph_topology_view",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/costs:graph_properties",
        "//machina/core/grappler/utils:canonicalizer",
        "//machina/core/grappler/utils:symbolic_shapes",
        "//machina/core/grappler/utils:topological_sort",
        "//machina/core/grappler/utils:traversal",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "arithmetic_optimizer_test_utils",
    testonly = 1,
    hdrs = [
        "arithmetic_optimizer_test_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":arithmetic_optimizer",
        ":common_subgraph_elimination",
        ":constant_folding",
        ":model_pruner",
        "//machina/core:test",
        "//machina/core/grappler/utils:grappler_test",
    ],
)

tf_cuda_cc_test(
    name = "arithmetic_optimizer_test",
    size = "small",
    srcs = ["arithmetic_optimizer_test.cc"],
    deps = [
        ":arithmetic_optimizer",
        ":arithmetic_optimizer_test_utils",
        ":model_pruner",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/cc:resource_variable_ops",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "common_subgraph_elimination",
    srcs = ["common_subgraph_elimination.cc"],
    hdrs = [
        "common_subgraph_elimination.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:graph_topology_view",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/utils:canonicalizer",
        "//machina/core/grappler/utils:topological_sort",
        "//machina/core/grappler/utils:traversal",
        "//machina/core/platform:hash",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cuda_cc_test(
    name = "common_subgraph_elimination_test",
    size = "small",
    srcs = ["common_subgraph_elimination_test.cc"],
    deps = [
        ":arithmetic_optimizer_test_utils",
        ":common_subgraph_elimination",
        ":model_pruner",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//machina/core/grappler/utils:grappler_test",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dependency_optimizer",
    srcs = ["dependency_optimizer.cc"],
    hdrs = [
        "dependency_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":constant_folding",
        ":graph_optimizer",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/costs:graph_properties",
        "//machina/core/grappler/utils:topological_sort",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cuda_cc_test(
    name = "dependency_optimizer_test",
    size = "small",
    srcs = ["dependency_optimizer_test.cc"],
    deps = [
        ":constant_folding",
        ":dependency_optimizer",
        ":model_pruner",
        "//machina/cc:cc_ops",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//machina/core/grappler/utils:grappler_test",
        "//machina/core/grappler/utils:topological_sort",
        "//machina/core/platform:test",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "model_pruner",
    srcs = ["model_pruner.cc"],
    hdrs = [
        "model_pruner.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:mutable_graph_view",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/utils:transitive_fanin",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cuda_cc_test(
    name = "model_pruner_test",
    srcs = ["model_pruner_test.cc"],
    deps = [
        ":model_pruner",
        "//machina/cc:cc_ops",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:devices",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//machina/core/grappler/utils:grappler_test",
    ],
)

tf_kernel_library(
    name = "gpu_swapping_kernels",
    srcs = [
        "gpu_swapping_kernels.cc",
    ],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "gpu_swapping_ops",
    srcs = [
        "gpu_swapping_ops.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "memory_optimizer",
    srcs = [
        "memory_optimizer.cc",
    ],
    hdrs = [
        "memory_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        ":static_schedule",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:graph_topology_view",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:mutable_graph_view",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/clusters:cluster",
        "//machina/core/grappler/clusters:virtual_cluster",
        "//machina/core/grappler/costs:graph_memory",
        "//machina/core/grappler/costs:graph_properties",
        "//machina/core/grappler/costs:utils",
        "//machina/core/grappler/utils:topological_sort",
        "//machina/core/grappler/utils:traversal",
    ],
)

tf_cuda_cc_test(
    name = "memory_optimizer_test",
    srcs = ["memory_optimizer_test.cc"],
    tags = [
        "no_cuda_on_cpu_tap",  # Do not re-enable again without actually testing.
        "no_windows",
    ],
    deps = [
        ":gpu_swapping_kernels",
        ":gpu_swapping_ops",
        ":memory_optimizer",
        "//machina/cc:cc_ops",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/clusters:virtual_cluster",
        "//machina/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "auto_mixed_precision",
    srcs = ["auto_mixed_precision.cc"],
    hdrs = [
        "auto_mixed_precision.h",
        "auto_mixed_precision_lists.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":custom_graph_optimizer_registry",
        ":graph_optimizer",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:devices",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:mutable_graph_view",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/clusters:cluster",
        "//machina/core/grappler/costs:virtual_placer",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cuda_cc_test(
    name = "auto_mixed_precision_test",
    srcs = ["auto_mixed_precision_test.cc"],
    deps = [
        ":auto_mixed_precision",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:devices",
        "//machina/core/grappler:graph_view",
        "//machina/core/grappler/clusters:single_machine",
        "//machina/core/grappler/clusters:virtual_cluster",
        "//machina/core/grappler/utils:grappler_test",
        "//machina/core/lib/random",
    ],
)

cc_library(
    name = "meta_optimizer",
    srcs = ["meta_optimizer.cc"],
    hdrs = [
        "meta_optimizer.h",
    ],
    copts = tf_copts(),
    defines = select({
        "//machina:fuchsia": ["NO_LLVM_SUPPORT=1"],
        "//machina:ios_x86_64": ["NO_LLVM_SUPPORT=1"],
        "//conditions:default": ["NO_LLVM_SUPPORT=0"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":arithmetic_optimizer",
        ":auto_mixed_precision",
        ":auto_parallel",
        ":common_subgraph_elimination",
        ":constant_folding",
        ":custom_graph_optimizer_registry",
        ":debug_stripper",
        ":dependency_optimizer",
        ":function_optimizer",
        ":generic_layout_optimizer",
        ":graph_optimizer",
        ":implementation_selector",
        ":loop_optimizer",
        ":memory_optimizer",
        ":model_pruner",
        ":pin_to_host_optimizer",
        ":remapper",
        ":scoped_allocator_optimizer",
        ":shape_optimizer",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/clusters:virtual_cluster",
        "//machina/core/grappler/utils:canonicalizer",
        "//machina/core/grappler/utils:colocation",
        "//machina/core/grappler/utils:functions",
        "//machina/core/grappler/utils:topological_sort",
        "//machina/core/grappler/utils:tpu",
        "//machina/core/grappler/verifiers:graph_verifier",
        "//machina/core/grappler/verifiers:structure_verifier",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ] + select({
        #TODO(b/200087693): LLVM does not build on Fuchsia.
        "//machina:fuchsia": [],
        # LLVM does not build on most apple platforms.
        "//machina:ios_x86_64": [],
        "//conditions:default": [":tfg_optimizer_hook"],
    }),
)

tf_cuda_cc_test(
    name = "meta_optimizer_test",
    srcs = ["meta_optimizer_test.cc"],
    tags = [
        "no_gpu",
    ],
    deps = [
        ":custom_graph_optimizer",
        ":custom_graph_optimizer_registry",
        ":meta_optimizer",
        "//machina/cc:cc_ops",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//machina/core/grappler/utils:grappler_test",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tfg_optimizer_hook",
    srcs = [
        "tfg_optimizer_hook.cc",
        "tfg_passes_builder.cc",
    ],
    hdrs = [
        "tfg_optimizer_hook.h",
        "tfg_passes_builder.h",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":graph_optimizer",
        "//machina/c:tf_status_headers",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/ir:Dialect",
        "//machina/core/ir:tf_op_registry",
        "//machina/core/ir/importexport:graphdef_export",
        "//machina/core/ir/importexport:graphdef_import",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/core/transforms:PassRegistration",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "tfg_optimizer_hook_test",
    srcs = ["tfg_optimizer_hook_test.cc"],
    deps = [
        ":tfg_optimizer_hook",
        "//machina/cc:const_op",
        "//machina/cc:scope",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/grappler:grappler_item",
        "@toolchain-project//mlir:Pass",
    ],
)

# This rule is header-only unless the build is static (--config=monolithic). Its
# implementation is included directly in the framework shared object.
cc_library(
    name = "custom_graph_optimizer_registry",
    hdrs = ["custom_graph_optimizer_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":custom_graph_optimizer",
        "//machina/core:lib",
    ] + if_static(
        [":custom_graph_optimizer_registry_impl"],
    ),
)

# This rule contains static variables for the optimizer registry. Do not depend
# on it directly; use :custom_graph_optimizer_registry, and link against
# libmachina_framework.so for the registry symbols.
cc_library(
    name = "custom_graph_optimizer_registry_impl",
    srcs = ["custom_graph_optimizer_registry.cc"],
    hdrs = ["custom_graph_optimizer_registry.h"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        ":custom_graph_optimizer",
        "//machina/core:lib",
    ],
)

tf_cuda_cc_test(
    name = "custom_graph_optimizer_registry_test",
    size = "small",
    srcs = ["custom_graph_optimizer_registry_test.cc"],
    deps = [
        ":custom_graph_optimizer",
        ":custom_graph_optimizer_registry",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "loop_optimizer",
    srcs = ["loop_optimizer.cc"],
    hdrs = [
        "loop_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":constant_folding",
        ":evaluation_utils",
        ":graph_optimizer",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:graph_topology_view",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:mutable_graph_view",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/costs:graph_properties",
        "//machina/core/grappler/utils:frame",
        "//machina/core/grappler/utils:traversal",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "loop_optimizer_test",
    srcs = ["loop_optimizer_test.cc"],
    deps = [
        ":loop_optimizer",
        "//machina/cc:cc_ops",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//machina/core/grappler/utils:graph_view",
        "//machina/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "shape_optimizer",
    srcs = ["shape_optimizer.cc"],
    hdrs = [
        "shape_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:mutable_graph_view",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/costs:graph_properties",
        "//machina/core/grappler/utils:frame",
        "//machina/core/grappler/utils:symbolic_shapes",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "shape_optimizer_test",
    srcs = ["shape_optimizer_test.cc"],
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":shape_optimizer",
        "//machina/cc:cc_ops",
        "//machina/core:core_cpu",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/utils:grappler_test",
    ],
)

tf_kernel_library(
    name = "remapper",
    srcs = ["remapper.cc"],
    hdrs = [
        "remapper.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":constant_folding",
        ":graph_optimizer",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:graph_view",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/costs:graph_properties",
        "//machina/core/grappler/utils:graph_view",
        "//machina/core/grappler/utils:pattern_utils",
        "//machina/core/grappler/utils:symbolic_shapes",
        "//machina/core/grappler/utils:topological_sort",
        "@com_google_absl//absl/container:flat_hash_set",
    ] + if_mkl(["//machina/core/graph:mkl_graph_util"]),
)

tf_cuda_cc_test(
    name = "remapper_test",
    srcs = ["remapper_test.cc"],
    tags = [],
    deps = [
        ":remapper",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:devices",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//machina/core/grappler/utils:grappler_test",
    ],
)

tf_cc_test_mkl(
    name = "mkl_remapper_test",
    srcs = ["mkl_remapper_test.cc"],
    tags = [
        "no_mac",
    ],
    deps = [
        ":remapper",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:devices",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//machina/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "debug_stripper",
    srcs = ["debug_stripper.cc"],
    hdrs = [
        "debug_stripper.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/clusters:cluster",
        "//machina/core/grappler/optimizers:graph_optimizer",
    ],
)

tf_cuda_cc_test(
    name = "debug_stripper_test",
    size = "small",
    srcs = ["debug_stripper_test.cc"],
    deps = [
        ":debug_stripper",
        "//machina/cc:cc_ops",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "scoped_allocator_optimizer",
    srcs = ["scoped_allocator_optimizer.cc"],
    hdrs = [
        "scoped_allocator_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/costs:graph_properties",
        "//machina/core/grappler/utils:frame",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "scoped_allocator_optimizer_test",
    size = "small",
    srcs = ["scoped_allocator_optimizer_test.cc"],
    deps = [
        ":scoped_allocator_optimizer",
        "//machina/cc:cc_ops",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu",
        "//machina/core:direct_session",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//machina/core/grappler/utils:topological_sort",
    ],
)

cc_library(
    name = "evaluation_utils",
    srcs = ["evaluation_utils.cc"],
    hdrs = [
        "evaluation_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "evaluation_utils_test",
    srcs = ["evaluation_utils_test.cc"],
    deps = [
        ":evaluation_utils",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "function_api_info",
    srcs = ["function_api_info.cc"],
    hdrs = ["function_api_info.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "function_api_info_test",
    size = "small",
    srcs = ["function_api_info_test.cc"],
    deps = [
        ":function_api_info",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "implementation_selector",
    srcs = ["implementation_selector.cc"],
    hdrs = ["implementation_selector.h"],
    deps = [
        ":custom_graph_optimizer",
        ":custom_graph_optimizer_registry",
        ":function_api_info",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/costs:graph_properties",
        "//machina/core/grappler/utils:graph_view",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "implementation_selector_test",
    size = "small",
    srcs = ["implementation_selector_test.cc"],
    deps = [
        ":custom_graph_optimizer",
        ":custom_graph_optimizer_registry",
        ":function_api_info",
        ":implementation_selector",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//machina/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "pin_to_host_optimizer",
    srcs = ["pin_to_host_optimizer.cc"],
    hdrs = [
        "pin_to_host_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:graph_view",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler/costs:graph_properties",
        "//machina/core/grappler/utils:symbolic_shapes",
        "//machina/core/grappler/utils:topological_sort",
        "//machina/core/grappler/utils:tpu",
    ],
)

tf_cuda_cc_test(
    name = "pin_to_host_optimizer_test",
    srcs = ["pin_to_host_optimizer_test.cc"],
    deps = [
        ":pin_to_host_optimizer",
        "//machina/cc:cc_ops",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/utils:grappler_test",
    ],
)

cc_library(
    name = "generic_layout_optimizer",
    srcs = ["generic_layout_optimizer.cc"],
    hdrs = ["generic_layout_optimizer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":generic_layout_optimizer_transposer",
        ":generic_layout_optimizer_transposer_factory",
        ":graph_optimizer",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler/clusters:cluster",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ] + if_static(
        ["//machina/core/platform:tensor_float_32_utils"],
        ["//machina/core/platform:tensor_float_32_hdr_lib"],
    ) + if_rocm_is_configured([
        "//machina/core/platform:stream_executor",
    ]),
)

tf_cuda_cc_test(
    name = "generic_layout_optimizer_test",
    size = "small",
    srcs = ["generic_layout_optimizer_test.cc"],
    deps = [
        ":generic_layout_optimizer",
        "//machina/cc:cc_ops",
        "//machina/core:framework",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/grappler:devices",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/clusters:cluster",
        "//machina/core/grappler/clusters:single_machine",
        "//machina/core/grappler/clusters:virtual_cluster",
        "//machina/core/grappler/utils:graph_view",
        "//machina/core/grappler/utils:grappler_test",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "generic_layout_optimizer_transposer",
    srcs = ["generic_layout_optimizer_transposer.cc"],
    hdrs = ["generic_layout_optimizer_transposer.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/costs:graph_properties",
        "//machina/core/grappler/utils:frame",
        "//machina/core/grappler/utils:graph_view",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cuda_cc_test(
    name = "generic_layout_optimizer_transposer_test",
    size = "small",
    srcs = ["generic_layout_optimizer_transposer_test.cc"],
    deps = [
        ":generic_layout_optimizer_transposer",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/grappler:devices",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/clusters:cluster",
        "//machina/core/grappler/clusters:single_machine",
        "//machina/core/grappler/clusters:virtual_cluster",
        "//machina/core/grappler/utils:graph_view",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "generic_layout_optimizer_transposer_factory",
    srcs = ["generic_layout_optimizer_transposer_factory.cc"],
    hdrs = ["generic_layout_optimizer_transposer_factory.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":generic_layout_optimizer_transposer",
        "//machina/core/grappler:op_types",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "generic_layout_optimizer_transposer_factory_test",
    srcs = ["generic_layout_optimizer_transposer_factory_test.cc"],
    deps = [
        ":generic_layout_optimizer_transposer",
        ":generic_layout_optimizer_transposer_factory",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
    ],
)
