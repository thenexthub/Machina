# Experimental extensions to the C API for eager execution of kernels.

load(
    "//machina:machina.bzl",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_cc_test",
    "tf_cuda_library",
)
load("//machina:machina.default.bzl", "cc_header_only_library", "filegroup")
load(
    "//machina/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_cuda_library(
    name = "c_api",
    srcs = [
        "c_api.cc",
        "c_api_debug.cc",
        "c_api_experimental.h",
        "c_api_internal.h",
        "c_api_unified_experimental.h",
    ],
    hdrs = ["c_api.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            ":immediate_execution_context",
            ":immediate_execution_distributed_manager",
            ":immediate_execution_operation",
            ":immediate_execution_tensor_handle",
            ":tfe_cancellation_manager_internal",
            ":tfe_context_internal",
            ":tfe_executor_internal",
            ":tfe_monitoring_internal",
            ":tfe_op_attrs_internal",
            ":tfe_op_internal",
            ":tfe_tensor_debug_info_internal",
            ":tfe_tensorhandle_internal",
            "//machina/c:c_api",
            "//machina/c:c_api_internal",
            "//machina/c:tf_buffer",
            "//machina/c:tf_buffer_internal",
            "//machina/c:tf_status_internal",
            "//machina/c:tf_tensor_internal",
            "//machina/core:core_cpu",
            "//machina/core:core_cpu_internal",
            "//machina/core:framework",
            "//machina/core:framework_internal",
            "//machina/core:lib",
            "//machina/core:lib_internal",
            "//machina/core:protos_all_cc",
            "//machina/core/common_runtime/eager:attr_builder",
            "//machina/core/common_runtime/eager:context",
            "//machina/core/common_runtime/eager:context_distributed_manager",
            "//machina/core/common_runtime/eager:core",
            "//machina/core/common_runtime/eager:custom_device",
            "//machina/core/common_runtime/eager:eager_executor",
            "//machina/core/common_runtime/eager:execute",
            "//machina/core/common_runtime/eager:placement_utils",
            "//machina/core/common_runtime/eager:tensor_handle",
            "//machina/core/profiler/lib:traceme",
            "@com_google_absl//absl/algorithm:container",
            "@com_google_absl//absl/types:span",
            "@com_google_absl//absl/types:variant",
        ],
    }) + [
        ":abstract_tensor_handle",
        "//machina/c:c_api_macros_hdrs",
        "//machina/core:gpu_runtime",
        "//machina/core/common_runtime/eager:eager_operation",
        "//machina/core/distributed_runtime:remote_device",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/distributed_runtime:worker_env",
        "//machina/core/distributed_runtime:worker_interface",
        "//machina/core/distributed_runtime/eager:cluster_function_library_runtime",
        "//machina/core/distributed_runtime/eager:eager_client",
        "//machina/core/distributed_runtime/eager:remote_mgr",
        "//machina/core/distributed_runtime/rpc:grpc_channel",
        "//machina/core/distributed_runtime/rpc:grpc_server_lib",
        "//machina/core/distributed_runtime/rpc:grpc_worker_cache",
        "//machina/core/distributed_runtime/rpc:grpc_worker_service",
        "//machina/core/distributed_runtime/rpc:rpc_rendezvous_mgr",
        "//machina/core/distributed_runtime/rpc/eager:grpc_eager_client",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:str_format",
        "@local_xla//xla/tsl/c:tsl_status_internal",
    ],
    alwayslink = 1,
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "abstract_context.h",
        "abstract_function.h",
        "abstract_op_attrs.h",
        "abstract_operation.h",
        "abstract_tensor_handle.h",
        "c_api.h",
        "c_api_experimental.h",
        "c_api_internal.h",
        "c_api_unified_experimental.h",
        "c_api_unified_experimental_internal.h",
        "dlpack.h",
        "gradients.h",
        "gradients_internal.h",
        "immediate_execution_context.h",
        "immediate_execution_distributed_manager.h",
        "immediate_execution_operation.h",
        "immediate_execution_tensor_handle.h",
        "tape.h",
        "tfe_cancellation_manager_internal.h",
        "tfe_context_internal.h",
        "tfe_executor_internal.h",
        "tfe_monitoring_internal.h",
        "tfe_op_attrs_internal.h",
        "tfe_tensor_debug_info_internal.h",
        "tfe_tensorhandle_internal.h",
    ],
    visibility = [
        "//machina/core/function/runtime_client:__pkg__",
        "//machina/python:__subpackages__",
    ],
)

cc_library(
    name = "c_api_internal",
    hdrs = [
        "c_api_experimental.h",
        "c_api_internal.h",
    ],
    visibility = ["//machina:internal"],
    deps = [
        ":c_api",
        ":tfe_cancellation_manager_internal",
        ":tfe_executor_internal",
        ":tfe_monitoring_internal",
        ":tfe_op_attrs_internal",
        ":tfe_tensor_debug_info_internal",
        "//machina/c:c_api_internal",
    ],
)

cc_library(
    name = "c_api_unified_internal",
    hdrs = [
        "c_api_unified_experimental_internal.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":abstract_context",
        ":abstract_operation",
        ":abstract_tensor_handle",
        ":c_api",
        ":c_api_experimental",
        "//machina/c:c_api_internal",
        "//machina/c:conversion_macros",
        "//machina/c:tf_status",
        "//machina/core:framework",
        "//machina/core/platform:casts",
        "//machina/core/platform:types",
    ],
)

cc_library(
    name = "tracing_utils",
    srcs = ["tracing_utils.cc"],
    hdrs = [
        "tracing_utils.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":abstract_operation",
        ":c_api_unified_internal",
        "//machina/c/experimental/gradients/tape:tape_operation",
        "//machina/core/lib/llvm_rtti",
        "//machina/core/platform:errors",
    ],
)

cc_library(
    name = "gradients_internal",
    srcs = [
        "gradients.cc",
    ],
    hdrs = [
        "gradients.h",
        "gradients_internal.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":abstract_context",
        ":abstract_tensor_handle",
        ":c_api_unified_internal",
        ":tape",
        "//machina/core/common_runtime/eager:attr_builder",
        "//machina/core/lib/llvm_rtti",
        "//machina/core/platform:errors",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "unified_api_testutil",
    testonly = 1,
    srcs = [
        "unified_api_testutil.cc",
    ],
    hdrs = [
        "unified_api_testutil.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":abstract_context",
        ":abstract_tensor_handle",
        ":c_api_experimental",
        ":c_api_test_util",
        ":c_api_unified_internal",
        "//machina/c:tf_status",
        "//machina/c:tf_status_helper",
        "//machina/c:tf_tensor",
        "//machina/core:framework",
        "//machina/core/lib/llvm_rtti",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cuda_cc_test(
    name = "gradients_test",
    size = "small",
    srcs = [
        "gradients_test.cc",
    ],
    args = ["--heap_check="],
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        ":abstract_context",
        ":abstract_tensor_handle",
        ":c_api_experimental",
        ":c_api_test_util",
        ":c_api_unified_internal",
        ":gradients_internal",
        ":unified_api_testutil",
        "//machina/c:c_api",
        "//machina/c:c_test_util",
        "//machina/c:tf_status_helper",
        "//machina/c/experimental/gradients:array_grad",
        "//machina/c/experimental/gradients:math_grad",
        "//machina/c/experimental/gradients:not_differentiable",
        "//machina/c/experimental/gradients/tape:tape_context",
        "//machina/c/experimental/ops",
        "//machina/compiler/mlir/machina/c:mlir_c_api_registration",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/lib/llvm_rtti",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cuda_cc_test(
    name = "unified_api_test",
    size = "small",
    srcs = [
        "unified_api_test.cc",
    ],
    args = ["--heap_check="],
    tags = tf_cuda_tests_tags() + ["no_cuda_asan"],  # b/173654156
    deps = [
        ":c_api_experimental",
        ":c_api_unified_internal",
        ":unified_api_testutil",
        "//machina/c:tf_status_helper",
        "//machina/compiler/mlir/machina/c:mlir_c_api_registration",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/lib/llvm_rtti",
        "//machina/core/platform:errors",
    ],
)

cc_library(
    name = "gradient_checker",
    testonly = 1,
    srcs = [
        "gradient_checker.cc",
    ],
    hdrs = [
        "gradient_checker.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":abstract_tensor_handle",
        ":unified_api_testutil",
        "//machina/c:tf_tensor_internal",
        "//machina/c/experimental/ops:math_ops",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cuda_cc_test(
    name = "gradient_checker_test",
    size = "small",
    srcs = [
        "gradient_checker_test.cc",
    ],
    args = ["--heap_check="],
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # b/175330074
    ],
    deps = [
        ":abstract_tensor_handle",
        ":c_api_experimental",
        ":gradient_checker",
        ":unified_api_testutil",
        "//machina/c:tf_status_helper",
        "//machina/c:tf_tensor_internal",
        "//machina/c/experimental/ops",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:tensor_float_32_utils",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "abstract_tensor_handle",
    srcs = ["abstract_tensor_handle.cc"],
    hdrs = ["abstract_tensor_handle.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:framework",
            "//machina/core:protos_all_cc",
            "//machina/core/platform:refcount",
            "//machina/core/platform:status",
        ],
    }),
)

cc_library(
    name = "immediate_execution_tensor_handle",
    srcs = ["immediate_execution_tensor_handle.cc"],
    hdrs = ["immediate_execution_tensor_handle.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":abstract_tensor_handle",
        "//machina/c:tensor_interface",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ],
)

cc_library(
    name = "abstract_op_attrs",
    hdrs = ["abstract_op_attrs.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "abstract_operation",
    hdrs = ["abstract_operation.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":abstract_tensor_handle",
        "//machina/c:tensor_interface",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "immediate_execution_operation",
    hdrs = ["immediate_execution_operation.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":abstract_operation",
        ":abstract_tensor_handle",
        ":immediate_execution_tensor_handle",
        "//machina/c:tensor_interface",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/util:managed_stack_trace",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "abstract_context",
    hdrs = ["abstract_context.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":abstract_function",
        ":abstract_operation",
    ],
)

cc_library(
    name = "abstract_function",
    hdrs = ["abstract_function.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        "//machina/core:protos_all_cc",
        "//machina/core/platform:intrusive_ptr",
        "//machina/core/platform:refcount",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "immediate_execution_distributed_manager",
    hdrs = ["immediate_execution_distributed_manager.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ],
)

cc_library(
    name = "immediate_execution_context",
    hdrs = ["immediate_execution_context.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":abstract_context",
        ":immediate_execution_distributed_manager",
        ":immediate_execution_operation",
        ":immediate_execution_tensor_handle",
        "//machina/c:tensor_interface",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tfe_context_internal",
    hdrs = ["tfe_context_internal.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":immediate_execution_context",
        "//machina/c:conversion_macros",
    ],
)

cc_library(
    name = "tfe_cancellation_manager_internal",
    hdrs = ["tfe_cancellation_manager_internal.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        "//machina/c:conversion_macros",
        "//machina/core:framework",
    ],
)

cc_library(
    name = "tfe_executor_internal",
    hdrs = ["tfe_executor_internal.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/core/common_runtime/eager:eager_executor",
    ],
)

cc_library(
    name = "tfe_monitoring_internal",
    hdrs = ["tfe_monitoring_internal.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/core:lib",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "tfe_op_attrs_internal",
    hdrs = ["tfe_op_attrs_internal.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":abstract_op_attrs",
        "//machina/c:conversion_macros",
        "//machina/c:tf_status",
        "//machina/core:protos_all_cc",
    ],
)

cc_library(
    name = "tfe_op_internal",
    hdrs = ["tfe_op_internal.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":immediate_execution_operation",
        "//machina/c:conversion_macros",
    ],
)

cc_library(
    name = "tfe_tensor_debug_info_internal",
    hdrs = ["tfe_tensor_debug_info_internal.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/core:lib",
    ],
)

cc_library(
    name = "tfe_tensorhandle_internal",
    hdrs = ["tfe_tensorhandle_internal.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":immediate_execution_tensor_handle",
        "//machina/c:conversion_macros",
    ],
)

cc_header_only_library(
    name = "tfe_tensorhandle_internal_hdrs_only",
    extra_deps = [
        "@com_google_absl//absl/strings",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":tfe_tensorhandle_internal",
    ],
)

cc_header_only_library(
    name = "tfe_cancellationmanager_internal_hdrs_only",
    extra_deps = [
        "@com_google_absl//absl/strings",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":tfe_cancellation_manager_internal",
    ],
)

tf_cuda_library(
    name = "c_api_test_util",
    testonly = 1,
    srcs = ["c_api_test_util.cc"],
    hdrs = ["c_api_test_util.h"],
    visibility = [
        "//learning/brain:__subpackages__",
        "//machina:__subpackages__",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_internal",
        "//machina/c:c_test_util",
        "//machina/c:tf_datatype",
        "//machina/c:tf_tensor",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
    ],
)

tf_cuda_cc_test(
    name = "c_api_test",
    size = "medium",
    srcs = [
        "c_api_debug_test.cc",
        "c_api_test.cc",
    ],
    tags = [
        "no_cuda_asan",  # TODO(b/181771536)
        "guitar",
        # "multi_gpu",  b/180748118
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_internal",
        ":c_api_test_util",
        ":tfe_op_internal",
        ":tfe_tensorhandle_internal",
        "//machina/c:c_test_util",
        "//machina/c:tf_datatype",
        "//machina/c:tf_status",
        "//machina/c:tf_tensor",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime/eager:eager_operation",
        "//machina/core/common_runtime/eager:tensor_handle",
        "//machina/core/distributed_runtime/rpc:grpc_server_lib",
        "//machina/core/platform:env",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_library(
    name = "c_api_remote_test_util",
    testonly = 1,
    srcs = ["c_api_remote_test_util.cc"],
    hdrs = ["c_api_remote_test_util.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":c_api",
        ":c_api_internal",
        ":c_api_test_util",
        ":tfe_tensorhandle_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core/common_runtime/eager:tensor_handle",
        "//machina/core/distributed_runtime/rpc:grpc_server_lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "c_api_remote_test",
    size = "medium",
    srcs = [
        "c_api_remote_test.cc",
    ],
    # TODO(b/136478427): Figure out how to correctly shut the server down
    args = ["--heap_check="],
    tags = [
        "no_windows",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_internal",
        ":c_api_remote_test_util",
        ":c_api_test_util",
        ":tfe_tensorhandle_internal",
        "//machina/c:c_test_util",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime:function_optimization_registry",
        "//machina/core/common_runtime/eager:eager_operation",
        "//machina/core/distributed_runtime/rpc:grpc_server_lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "c_api_remote_function_test",
    size = "medium",
    srcs = [
        "c_api_remote_function_test.cc",
    ],
    # TODO(b/136478427): Figure out how to correctly shut the server down
    args = ["--heap_check="],
    tags = [
        "no_windows",
    ],
    deps = [
        ":c_api_remote_test_util",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

tf_cuda_cc_test(
    name = "c_api_distributed_test",
    size = "small",
    srcs = [
        "c_api_distributed_test.cc",
    ],
    # TODO(b/136478427): Figure out how to correctly shut the server down
    args = ["--heap_check="],
    tags = [
        "no_oss",  # TODO(b/200848572)
        "no_windows",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_internal",
        ":c_api_test_util",
        ":tfe_tensorhandle_internal",
        "//machina/c:c_test_util",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime:function_optimization_registry",
        "//machina/core/common_runtime:optimization_registry",
        "//machina/core/common_runtime/eager:eager_operation",
        "//machina/core/distributed_runtime/rpc:grpc_server_lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "c_api_cluster_test",
    size = "medium",
    srcs = [
        "c_api_cluster_test.cc",
    ],
    # TODO(b/136478427): Figure out how to correctly shut the server down
    args = ["--heap_check="],
    tags = [
        "no_windows",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_internal",
        ":c_api_test_util",
        ":tfe_tensorhandle_internal",
        "//machina/c:c_test_util",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime/eager:eager_operation",
        "//machina/core/distributed_runtime/rpc:grpc_server_lib",
        "//machina/core/platform:env",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "graph_function",
    srcs = ["graph_function.cc"],
    hdrs = [
        "graph_function.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":abstract_function",
        "//machina/core:framework",
        "//machina/core/platform:refcount",
        "//machina/core/platform:status",
    ],
)

tf_cuda_library(
    name = "c_api_experimental_hdrs",
    hdrs = [
        "c_api.h",
        "c_api_experimental.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/c:c_api_headers",
        "//machina/c:c_api_macros_hdrs",
    ],
)

tf_cuda_library(
    name = "c_api_experimental",
    srcs = [
        "c_api_experimental.cc",
        "c_api_unified_experimental.cc",
        "c_api_unified_experimental_eager.cc",
        "c_api_unified_experimental_graph.cc",
        "c_api_unified_experimental_internal.h",
    ],
    hdrs = [
        "c_api_experimental.h",
        "c_api_unified_experimental.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            ":abstract_context",
            ":abstract_operation",
            ":abstract_tensor_handle",
            ":c_api",
            ":c_api_internal",
            ":graph_function",
            ":immediate_execution_context",
            ":immediate_execution_tensor_handle",
            ":tfe_context_internal",
            ":tfe_op_internal",
            ":tfe_tensorhandle_internal",
            "//machina/c:c_api",
            "//machina/c:c_api_internal",
            "//machina/c:conversion_macros",
            "//machina/core:core_cpu",
            "//machina/core:core_cpu_internal",
            "//machina/core:framework",
            "//machina/core:framework_internal",
            "//machina/core:lib",
            "//machina/core:lib_internal",
            "//machina/core:protos_all_cc",
            "//machina/core/common_runtime/eager:attr_builder",
            "//machina/core/common_runtime/eager:context",
            "//machina/core/common_runtime/eager:copy_to_device_node",
            "//machina/core/common_runtime/eager:eager_executor",
            "//machina/core/common_runtime/eager:eager_operation",
            "//machina/core/common_runtime/eager:execute",
            "//machina/core/common_runtime/eager:kernel_and_device",
            "//machina/core/common_runtime/eager:tensor_handle",
            "//machina/core/lib/llvm_rtti",
            "@com_google_absl//absl/types:variant",
        ],
    }) + select({
        "//machina:with_xla_support": [
            "//machina/compiler/jit",
            "//machina/compiler/jit:xla_device",
            "//machina/compiler/tf2xla:xla_compiler",
        ],
        "//conditions:default": [],
    }) + [
        "//machina/c:tf_status_helper",
        "//machina/core:gpu_runtime",
        "//machina/core/distributed_runtime:remote_device",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/distributed_runtime:worker_env",
        "//machina/core/distributed_runtime/coordination:coordination_service_error_util",
        "//machina/core/distributed_runtime/eager:eager_client",
        "//machina/core/distributed_runtime/rpc:grpc_channel",
        "//machina/core/distributed_runtime/rpc:grpc_server_lib",
        "//machina/core/distributed_runtime/rpc:grpc_worker_cache",
        "//machina/core/distributed_runtime/rpc:grpc_worker_service",
        "//machina/core/distributed_runtime/rpc:rpc_rendezvous_mgr",
        "//machina/core/distributed_runtime/rpc/eager:grpc_eager_client",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_xla//xla/tsl/c:tsl_status_internal",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
    ],
    alwayslink = 1,
)

cc_library(
    name = "c_api_experimental_reader",
    testonly = True,
    srcs = [
        "c_api_experimental_reader.cc",
    ],
    hdrs = [
        "c_api_experimental_reader.h",
        "tfe_monitoring_reader_internal.h",
    ],
    visibility = ["//machina:internal"],
    deps = [
        ":c_api",
        "//machina/c:c_api",
        "//machina/core/lib/monitoring:cell_reader",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "c_api_experimental_test",
    size = "small",
    srcs = [
        "c_api_experimental_test.cc",
    ],
    args = ["--heap_check="],
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_internal",
        ":c_api_test_util",
        "//machina/c:c_test_util",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/distributed_runtime/rpc:grpc_server_lib",
        "//machina/core/platform:status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "c_api_experimental_reader_test",
    size = "small",
    srcs = [
        "c_api_experimental_reader_test.cc",
    ],
    args = ["--heap_check="],
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_experimental_reader",
        ":c_api_test_util",
        "//machina/c:c_test_util",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_cc_test(
    name = "c_api_unified_experimental_test",
    size = "small",
    srcs = [
        "c_api_unified_experimental_internal.h",
        "c_api_unified_experimental_test.cc",
    ],
    args = ["--heap_check="],
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        ":abstract_context",
        ":abstract_operation",
        ":abstract_tensor_handle",
        ":c_api",
        ":c_api_experimental",
        ":c_api_test_util",
        "//machina/c:c_api",
        "//machina/c:c_test_util",
        "//machina/c:conversion_macros",
        "//machina/c:tf_status_helper",
        "//machina/compiler/mlir/machina/c:mlir_c_api_registration",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "custom_device_testutil",
    testonly = True,
    srcs = ["custom_device_testutil.cc"],
    hdrs = ["custom_device_testutil.h"],
    visibility = ["//machina:internal"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_test_util",
        "//machina/c:c_api",
        "//machina/core:lib",
        "//machina/core:test",
    ],
)

tf_cc_test(
    name = "custom_device_test",
    size = "medium",
    srcs = [
        "custom_device_test.cc",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_test_util",
        ":custom_device_testutil",
        "//machina/c:c_api",
        "//machina/c:c_test_util",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tape",
    hdrs = ["tape.h"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/config:flag_defs",
        "//machina/core/config:flags",
    ],
)

filegroup(
    name = "headers",
    srcs = [
        "c_api.h",
        "c_api_experimental.h",
        "dlpack.h",
    ],
    visibility = ["//machina:__subpackages__"],
)

filegroup(
    name = "pywrap_headers_monitoring_reader",
    srcs = [
        "c_api_experimental_reader.h",
        "tfe_monitoring_reader_internal.h",
    ],
    visibility = ["//machina:__subpackages__"],
)

filegroup(
    name = "headers_monitoring_reader",
    srcs = [
        "c_api_experimental_reader.h",
    ],
    visibility = ["//machina:__subpackages__"],
)

cc_library(
    name = "dlpack",
    srcs = ["dlpack.cc"],
    hdrs = ["dlpack.h"],
    copts = tf_copts() + [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":tfe_tensorhandle_internal",
        "//machina/c:tf_status_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core/common_runtime/eager:tensor_handle",
        "@dlpack",
    ],
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "dlpack_test",
    size = "medium",
    srcs = [
        "dlpack_test.cc",
    ],
    args = [],
    tags = [],
    deps = [
        ":c_api",
        ":dlpack",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:status",
        "@com_google_absl//absl/strings",
        "@dlpack",
    ],
)

# TODO(karllessard): only used by //machina/core:mobile_srcs_only_runtime
# right now, remove this public rule when no longer needed (it should be
# replaced by TF Lite)
filegroup(
    name = "srcs",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "c_api_experimental.cc",
            "c_api_experimental_reader.cc",
            "c_api_experimental_reader.h",
            "tfe_monitoring_reader_internal.h",
            "c_api_unified_experimental.cc",
            "c_api_unified_experimental_eager.cc",
            "c_api_unified_experimental_graph.cc",
            "c_api_unified_experimental_internal.h",
            "graph_function.h",
            "graph_function.cc",
            "gradient_checker.cc",
            "gradient_checker.h",
            "gradients.cc",  # Uses RTTI.
            "tracing_utils.h",
            "tracing_utils.cc",
            "*test*",
            "*dlpack*",
        ],
    ),
    visibility = ["//visibility:public"],
)
