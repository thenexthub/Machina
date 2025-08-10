load("@bazel_skylib//lib:selects.bzl", "selects")
load(
    "@local_xla//xla/tsl/mkl:build_defs.bzl",
    "if_mkl",
    "mkl_deps",
)
load(
    "//machina:machina.bzl",
    "clean_dep",
    "if_google",
    "if_zendnn",
    "tf_cc_test",
    "tf_cc_test_mkl",
    "tf_copts",
    "tf_cuda_library",
    "tf_mkl_kernel_library",
)
load("//machina:machina.default.bzl", "filegroup")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

# TODO(b/152902651): Remove this file once all circular dependencies are resolved.
tf_cuda_library(
    name = "core",
    visibility = ["//machina:internal"],
    deps = [
        ":core_no_xla",
        "//machina/compiler/jit:xla_kernel_creator",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "core_no_xla",
    srcs = [
        "core.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":context",
        ":eager_operation",
        ":execute",
        ":placement_utils",
        ":tensor_handle",
        "//machina/c:c_api_internal",
        "//machina/c:tf_tensor_internal",
        "//machina/c/eager:abstract_function",
        "//machina/core/platform:errors",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "eager_executor",
    srcs = [
        "eager_executor.cc",
    ],
    hdrs = [
        "eager_executor.h",
    ],
    visibility = ["//machina:internal"],
    deps = select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:core_cpu_lib",
            "//machina/core:framework",
            "//machina/core:framework_internal",
            "//machina/core:lib",
            "//machina/core:lib_internal",
            "//machina/core:protos_all_cc",
            "@com_google_absl//absl/container:flat_hash_map",
        ],
    }),
)

tf_cc_test(
    name = "eager_executor_test",
    srcs = ["eager_executor_test.cc"],
    deps = [
        ":core",
        ":eager_executor",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:status_matchers",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cuda_library(
    name = "context",
    srcs = [
        "context.cc",
    ],
    hdrs = [
        "context.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":custom_device",
        ":eager_executor",
        ":kernel_and_device",
        ":rendezvous_cache",
        ":small_constants_optimizer",
        ":summary_optimizer",
        "//machina/c:tensor_interface",
        "//machina/c:tf_tensor_internal",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_distributed_manager",
        "//machina/c/eager:immediate_execution_operation",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/core/distributed_runtime:rendezvous_mgr_interface",
        "//machina/core/distributed_runtime:worker_env",
        "//machina/core/nccl:collective_communicator",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_xla//xla/tsl/util:env_var",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:core_cpu_lib",
            "//machina/core:framework",
            "//machina/core:framework_internal",
            "//machina/core:lib",
            "//machina/core:lib_internal",
            "//machina/core:protos_all_cc",
            "//machina/core:session_options",
            "//machina/core/distributed_runtime:cluster_function_library_runtime",
            "//machina/core/distributed_runtime:collective_param_resolver_distributed",
            "//machina/core/distributed_runtime:device_resolver_distributed",
            "//machina/core/distributed_runtime:rpc_collective_executor_mgr",
            "//machina/core/distributed_runtime:server_lib",
            "//machina/core/distributed_runtime:session_mgr",
            "//machina/core/distributed_runtime:worker_cache",
            "//machina/core/distributed_runtime:worker_session",
            "//machina/core/distributed_runtime/eager:eager_client",
            "@com_google_absl//absl/types:optional",
        ],
    }),
)

tf_cuda_library(
    name = "custom_device",
    srcs = [
        "custom_device.cc",
        "custom_device_op_handler.cc",
    ],
    hdrs = [
        "custom_device.h",
        "custom_device_op_handler.h",
    ],
    visibility = ["//machina:internal"],
    deps = select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/c/eager:immediate_execution_context",
            "//machina/c/eager:immediate_execution_operation",
            "//machina/c/eager:immediate_execution_tensor_handle",
            "//machina/core:framework",
            "//machina/core:lib",
            "//machina/core/framework:full_type_proto_cc",
            "//machina/core/lib/core:status",
        ],
    }),
)

tf_cc_test(
    name = "custom_device_test",
    srcs = ["custom_device_test.cc"],
    deps = [
        ":context",
        ":core",
        ":custom_device",
        ":eager_operation",
        ":placement_utils",
        ":tensor_handle",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:full_type_proto_cc",
        "//machina/core/kernels:resource_variable_ops",
        "//machina/core/ops:resource_variable_ops_op_lib",
    ],
)

tf_cuda_library(
    name = "context_distributed_manager",
    srcs = [
        "context_distributed_manager.cc",
    ],
    hdrs = [
        "context_distributed_manager.h",
    ],
    visibility = ["//machina:internal"],
    deps = [
        ":context",
        ":eager_executor",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_distributed_manager",
        "//machina/core/distributed_runtime:master_env",
        "//machina/core/distributed_runtime:rendezvous_mgr_interface",
        "//machina/core/distributed_runtime:worker_env",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:core_cpu_lib",
            "//machina/core:framework",
            "//machina/core:framework_internal",
            "//machina/core:lib",
            "//machina/core:lib_internal",
            "//machina/core:protos_all_cc",
            "//machina/core:session_options",
            "//machina/core/distributed_runtime:cluster_function_library_runtime",
            "//machina/core/distributed_runtime:remote_device",
            "//machina/core/distributed_runtime:server_lib",
            "//machina/core/distributed_runtime:session_mgr",
            "//machina/core/distributed_runtime:worker_cache",
            "//machina/core/distributed_runtime:worker_interface",
            "//machina/core/distributed_runtime:worker_session",
            "//machina/core/distributed_runtime/eager:cluster_function_library_runtime",
            "//machina/core/distributed_runtime/eager:eager_client",
            "//machina/core/distributed_runtime/eager:remote_mgr",
            "//machina/core/tfrt/common:global_state",
            "//machina/core/tfrt/common:pjrt_state",
            "//machina/core/tfrt/common:pjrt_util",
            "@com_google_absl//absl/base:core_headers",
            "@com_google_absl//absl/container:flat_hash_map",
            "@com_google_absl//absl/log",
            "@com_google_absl//absl/status",
            "@com_google_absl//absl/status:statusor",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/synchronization",
            "@com_google_absl//absl/time",
            "@local_tsl//tsl/platform:mutex",
            "@local_xla//xla/pjrt/gpu:se_gpu_pjrt_client",
            "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service",
            "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
            "@local_xla//xla/tsl/distributed_runtime/preemption:preemption_notifier",
            "@local_xla//xla/tsl/platform:statusor",
        ],
    }) + if_google(
        # TODO(b/282068262): PJRT pulls in TFRT components that are incompatible with ARM platform.
        # Clean up so that PJRT can run on ARM (and remove "#if defined(PLATFORM_GOOGLE) ..." use
        # from gpu_util.cc).
        # Also it won't build with WeightWatcher which tracks OSS build binaries.
        # TODO(b/290533709): Clean up this build rule.
        selects.with_or({
            clean_dep("//machina:linux_x86_64_with_weightwatcher"): [],
            (
                clean_dep("//machina:linux_x86_64"),
            ): [
                "//machina/core",
                "//machina/core/framework:resource_base",
                "@local_xla//xla/pjrt/distributed:key_value_store_interface",
                "@local_xla//xla/pjrt:local_device_state",
                "@local_xla//xla/pjrt/gpu:gpu_topology",
                "@local_xla//xla/pjrt:pjrt_client",
                "@local_xla//xla/pjrt:pjrt_compiler",
                "@local_xla//xla/service/gpu:gpu_executable_run_options",
                "//machina/core/common_runtime/gpu:gpu_runtime_impl",
            ],
            "//conditions:default": [],
        }),
    ),
)

tf_cc_test(
    name = "context_test",
    srcs = ["context_test.cc"],
    deps = [
        ":context",
        ":context_distributed_manager",
        ":core",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_operation",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:portable_gif_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime:composite_device",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/kernels:cast_op",
        "//machina/core/kernels:logging_ops",
        "//machina/core/kernels:math",
        "//machina/core/platform:refcount",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:status",
    ],
)

tf_cc_test(
    name = "placement_test",
    srcs = ["placement_test.cc"],
    deps = [
        ":context",
        ":core",
        ":eager_operation",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

tf_cuda_library(
    name = "eager_operation",
    srcs = [
        "eager_operation.cc",
    ],
    hdrs = [
        "eager_operation.h",
    ],
    visibility = ["//machina:internal"],
    deps = [
        ":attr_builder",
        ":context",
        ":custom_device",
        ":eager_executor",
        ":kernel_and_device",
        ":tensor_handle",
        "//machina/c:tf_tensor_internal",
        "//machina/c/eager:abstract_operation",
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:immediate_execution_operation",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:platform_port",
        "//machina/core/util:managed_stack_trace",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:core_cpu_lib",
        ],
    }),
)

tf_cc_test(
    name = "eager_operation_test",
    srcs = ["eager_operation_test.cc"],
    deps = [
        ":core",
        ":eager_operation",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

tf_cuda_library(
    name = "tensor_handle_data",
    srcs = [
        "tensor_handle_data.cc",
    ],
    hdrs = [
        "tensor_handle_data.h",
    ],
    visibility = ["//machina:internal"],
    deps = [
        ":context",
        ":eager_executor",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:framework",
            "//machina/core:lib",
            "//machina/core/profiler/lib:traceme",
            "@com_google_absl//absl/types:variant",
        ],
    }),
)

tf_cc_test(
    name = "tensor_handle_data_test",
    srcs = ["tensor_handle_data_test.cc"],
    deps = [
        ":core",
        ":tensor_handle_data",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:status_matchers",
    ],
)

tf_cuda_library(
    name = "tensor_handle",
    srcs = [
        "tensor_handle.cc",
    ],
    hdrs = [
        "tensor_handle.h",
    ],
    visibility = ["//machina:internal"],
    deps = [
        ":eager_executor",
        ":kernel_and_device",
        ":tensor_handle_data",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/c:tf_tensor_internal",
            "//machina/c/eager:immediate_execution_tensor_handle",
            "//machina/core:core_cpu_lib",
            "//machina/core:framework",
            "//machina/core:framework_internal",
            "//machina/core:lib",
            "//machina/core:lib_internal",
            "//machina/core:protos_all_cc",
            "//machina/core:session_options",
            "//machina/core/distributed_runtime/eager:remote_tensor_handle_data",
            "//machina/core/profiler/lib:traceme",
            "@com_google_absl//absl/status",
            "@com_google_absl//absl/strings",
            "@com_google_absl//absl/types:variant",
        ],
    }),
)

tf_cc_test(
    name = "tensor_handle_test",
    srcs = ["tensor_handle_test.cc"],
    deps = [
        ":core",
        ":tensor_handle",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:status_matchers",
        "@com_google_absl//absl/cleanup",
    ],
)

tf_cuda_library(
    name = "copy_to_device_node",
    hdrs = [
        "copy_to_device_node.h",
    ],
    visibility = ["//machina:internal"],
    deps = [
        ":context",
        ":eager_executor",
        ":tensor_handle",
    ] + select({
        "//machina:android": [
        ],
        "//conditions:default": [
            "//machina/core:core_cpu_lib",
            "//machina/core:framework",
            "//machina/core:framework_internal",
            "//machina/core:lib",
            "//machina/core:lib_internal",
            "//machina/core:protos_all_cc",
            "//machina/core:session_options",
            "//machina/core/profiler/lib:scoped_memory_debug_annotation",
        ],
    }),
)

cc_library(
    name = "shape_inference",
    srcs = ["shape_inference.cc"],
    hdrs = ["shape_inference.h"],
    visibility = ["//machina:internal"],
    deps = [
        ":tensor_handle",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ],
)

cc_library(
    name = "small_constants_optimizer",
    srcs = ["small_constants_optimizer.cc"],
    hdrs = ["small_constants_optimizer.h"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "summary_optimizer",
    srcs = ["summary_optimizer.cc"],
    hdrs = ["summary_optimizer.h"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "summary_optimizer_test",
    srcs = ["summary_optimizer_test.cc"],
    deps = [
        ":summary_optimizer",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_googletest//:gtest_main",
    ],
)

KERNEL_AND_DEVICE_DEPS = [
    "//machina/core:core_cpu_lib",
    "//machina/core:framework",
    "//machina/core:framework_internal",
    "//machina/core:lib",
    "//machina/core:lib_internal",
    "//machina/core:protos_all_cc",
    "//machina/core/profiler/lib:annotated_traceme",
    "//machina/core/profiler/lib:traceme",
    "//machina/core/grappler:grappler_item",
    "//machina/core/grappler/optimizers:meta_optimizer",
]

tf_cuda_library(
    name = "kernel_and_device",
    srcs = [
        "kernel_and_device.cc",
    ],
    hdrs = [
        "kernel_and_device.h",
    ],
    visibility = ["//machina:internal"],
    deps = [
        ":attr_builder",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@farmhash_archive//:farmhash",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": KERNEL_AND_DEVICE_DEPS,
    }),
)

tf_cc_test(
    name = "kernel_and_device_test",
    srcs = ["kernel_and_device_test.cc"],
    deps = [
        ":attr_builder",
        ":kernel_and_device",
        "//machina/cc:cc_ops",
        "//machina/cc:client_session",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/core:array_ops_op_lib",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:direct_session",
        "//machina/core:framework",
        "//machina/core:functional_ops_op_lib",
        "//machina/core:lib",
        "//machina/core:math_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core:no_op_op_lib",
        "//machina/core:protos_all_cc",
        "//machina/core:sendrecv_ops_op_lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/kernels:constant_op",
        "//machina/core/kernels:matmul_op",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "execute",
    srcs = [
        "execute.cc",
        "execute_node.cc",
    ],
    hdrs = [
        "execute.h",
        "execute_node.h",
    ],
    copts = if_mkl(["-DINTEL_MKL"]),
    deps = [
        ":context",
        ":copy_to_device_node",
        ":eager_executor",
        ":eager_op_rewrite_registry",
        ":eager_operation",
        ":kernel_and_device",
        ":small_constants_optimizer",
        ":summary_optimizer",
        ":tensor_handle",
        "//machina/c:tf_tensor_internal",
        "//machina/compiler/jit:common",
        "//machina/core/profiler/lib:scoped_memory_debug_annotation",
        "//machina/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/util:env_var",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:core_cpu_lib",
            "//machina/core:framework",
            "//machina/core:framework_internal",
            "//machina/core:lib",
            "//machina/core:lib_internal",
            "//machina/core:protos_all_cc",
            "//machina/core/distributed_runtime/eager:eager_client",
            "//machina/core/distributed_runtime/eager:remote_copy_node",
            "//machina/core/distributed_runtime/eager:remote_execute_node",
            "//machina/core/distributed_runtime/eager:remote_mgr",
        ],
    }) + if_mkl([
        ":mkl_eager_op_rewrite",
        "//machina/core/graph:mkl_graph_util",
    ]) + if_zendnn([
        ":zen_eager_op_rewrite",
        "//machina/core/graph:zen_graph_util",
    ]),
)

tf_cc_test(
    name = "execute_test",
    srcs = ["execute_test.cc"],
    deps = [
        ":core",
        ":execute",
        "//machina/compiler/jit:xla_cpu_jit",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:full_type_proto_cc",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/kernels:math",
        "//machina/core/kernels:partitioned_function_ops",
        "//machina/core/lib/monitoring:cell_reader",
    ],
)

tf_cc_test(
    name = "execute_node_test",
    srcs = ["execute_node_test.cc"],
    deps = [
        ":context",
        ":core",
        ":execute",
        ":kernel_and_device",
        ":tensor_handle",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "zen_eager_op_rewrite",
    srcs = ["zen_eager_op_rewrite.cc"],
    copts = tf_copts(),
    deps = [
        ":eager_op_rewrite_registry",
        "//machina/core:all_kernels",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core/graph:zen_graph_util",
    ],
    alwayslink = 1,
)

tf_mkl_kernel_library(
    name = "mkl_eager_op_rewrite",
    srcs = ["mkl_eager_op_rewrite.cc"],
    deps = [
        ":eager_op_rewrite_registry",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core/graph:mkl_graph_util",
    ] + mkl_deps(),
    alwayslink = 1,
)

tf_cc_test_mkl(
    name = "mkl_eager_op_rewrite_test",
    srcs = ["mkl_eager_op_rewrite_test.cc"],
    tags = [
        "nomac",
    ],
    deps = [
        ":core",
        ":eager_op_rewrite_registry",
        ":mkl_eager_op_rewrite",
        "//machina/core:all_kernels",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime:device_mgr",
    ],
)

cc_library(
    name = "eager_op_rewrite_registry",
    srcs = ["eager_op_rewrite_registry.cc"],
    hdrs = ["eager_op_rewrite_registry.h"],
    deps = [":eager_operation"],
)

tf_cc_test(
    name = "eager_op_rewrite_registry_test",
    srcs = ["eager_op_rewrite_registry_test.cc"],
    deps = [
        ":eager_op_rewrite_registry",
        ":mkl_core",
        "//machina/core:lib",
        "//machina/core:no_op_op_lib",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

# Temporary rule until the circular dependencies issue is resolved.
# TODO(mabuzain): remove this once original "core" package is fixed.
cc_library(
    name = "mkl_core",
    srcs = [
        "core.cc",
        "execute.cc",
        "execute_node.cc",
    ],
    hdrs = [
        "execute.h",
        "execute_node.h",
    ],
    copts = tf_copts(),
    deps = [
        ":context",
        ":copy_to_device_node",
        ":eager_executor",
        ":eager_op_rewrite_registry",
        ":eager_operation",
        ":kernel_and_device",
        ":placement_utils",
        ":small_constants_optimizer",
        ":summary_optimizer",
        ":tensor_handle",
        "//machina/c:c_api_internal",
        "//machina/c:tf_tensor_internal",
        "//machina/c/eager:abstract_function",
        "//machina/compiler/jit:common",
        "//machina/core/platform:errors",
        "//machina/core/profiler/lib:scoped_memory_debug_annotation",
        "//machina/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/util:env_var",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:core_cpu_lib",
            "//machina/core:framework",
            "//machina/core:framework_internal",
            "//machina/core:lib",
            "//machina/core:lib_internal",
            "//machina/core:protos_all_cc",
            "//machina/core/distributed_runtime/eager:eager_client",
            "//machina/core/distributed_runtime/eager:remote_copy_node",
            "//machina/core/distributed_runtime/eager:remote_execute_node",
            "//machina/core/distributed_runtime/eager:remote_mgr",
        ],
    }),
)

tf_cuda_library(
    name = "placement_utils",
    srcs = [
        "placement_utils.cc",
    ],
    hdrs = [
        "placement_utils.h",
    ],
    visibility = ["//machina:internal"],
    deps = [
        ":attr_builder",
        ":context",
        ":custom_device",
        ":eager_operation",
        "//machina/c/eager:immediate_execution_operation",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "@com_google_absl//absl/status",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:core_cpu_lib",
            "//machina/core:framework",
            "//machina/core:framework_internal",
            "//machina/core:lib",
            "//machina/core:lib_internal",
            "//machina/core:protos_all_cc",
        ],
    }),
)

tf_cc_test(
    name = "placement_utils_test",
    srcs = ["placement_utils_test.cc"],
    deps = [
        ":core",
        ":eager_operation",
        ":execute",
        ":placement_utils",
        ":tensor_handle",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime:device_mgr",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:array",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "rendezvous_cache",
    hdrs = ["rendezvous_cache.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:refcount",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

tf_cuda_library(
    name = "attr_builder",
    srcs = ["attr_builder.cc"],
    hdrs = ["attr_builder.h"],
    visibility = ["//machina:internal"],
    deps = [
        "@com_google_absl//absl/status",
        # Only the TF_AttrType enum is required, so pull in just the C headers.
        # TODO(b/113535673): Break this dependency and avoid the C header completely.
        "//machina/c:tf_attrtype",
        "@farmhash_archive//:farmhash",
        "//machina/c/eager:abstract_op_attrs",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:core_cpu",
            "//machina/core:core_cpu_internal",
            "//machina/core:framework",
            "//machina/core:framework_internal",
            "//machina/core:lib",
            "//machina/core:lib_internal",
            "//machina/core:protos_all_cc",
        ],
    }),
)

tf_cc_test(
    name = "attr_builder_test",
    srcs = ["attr_builder_test.cc"],
    deps = [
        ":attr_builder",
        "//machina/c:c_api",
        "//machina/cc:cc_ops",
        "//machina/cc:client_session",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:types_proto_cc",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "attr_builder.h",
        "context.h",
        "custom_device.h",
        "custom_device_op_handler.h",
        "eager_executor.h",
        "eager_operation.h",
        "kernel_and_device.h",
        "rendezvous_cache.h",
        "tensor_handle.h",
        "tensor_handle_data.h",
    ],
    visibility = [
        "//machina/core/function/runtime_client:__pkg__",
        "//machina/python:__subpackages__",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "*_test.cc",
            "remote_*",
        ],
    ),
)
