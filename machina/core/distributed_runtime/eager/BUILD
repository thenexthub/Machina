load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load("//machina:machina.default.bzl", "filegroup", "tf_grpc_cc_dependencies")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "remote_tensor_handle",
    hdrs = ["remote_tensor_handle.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ],
)

cc_library(
    name = "cluster_function_library_runtime",
    srcs = [
        "cluster_function_library_runtime.cc",
    ],
    hdrs = [
        "cluster_function_library_runtime.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":eager_client",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/common_runtime/eager:eager_operation",
        "//machina/core/distributed_runtime:call_options",
        "//machina/core/distributed_runtime:worker_session",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "destroy_tensor_handle_node",
    hdrs = ["destroy_tensor_handle_node.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":eager_client",
        "//machina/core/common_runtime/eager:eager_executor",
        "//machina/core/protobuf:eager_service_proto_cc",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "eager_client",
    hdrs = ["eager_client.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/distributed_runtime:call_options",
        "//machina/core/protobuf:eager_service_proto_cc",
    ],
)

cc_library(
    name = "remote_execute_node",
    srcs = ["remote_execute_node.cc"],
    hdrs = ["remote_execute_node.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":eager_client",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime/eager:eager_executor",
        "//machina/core/common_runtime/eager:shape_inference",
        "//machina/core/common_runtime/eager:tensor_handle",
        "//machina/core/protobuf:eager_service_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "eager_service_impl",
    srcs = ["eager_service_impl.cc"],
    hdrs = [
        "eager_service_impl.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":cluster_function_library_runtime",
        ":remote_mgr",
        ":remote_tensor_handle",
        "//machina/c/eager:immediate_execution_distributed_manager",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/common_runtime/eager:context_distributed_manager",
        "//machina/core/common_runtime/eager:core",
        "//machina/core/common_runtime/eager:eager_operation",
        "//machina/core/common_runtime/eager:execute",
        "//machina/core/distributed_runtime:message_wrappers",
        "//machina/core/distributed_runtime:rpc_collective_executor_mgr",
        "//machina/core/distributed_runtime:session_mgr",
        "//machina/core/distributed_runtime:worker_cache",
        "//machina/core/distributed_runtime:worker_env",
        "//machina/core/nccl:collective_communicator",
        "//machina/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla/tsl/distributed_runtime/preemption:preemption_notifier",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "eager_service_impl_test",
    srcs = ["eager_service_impl_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":cluster_function_library_runtime",
        ":eager_service_impl",
        ":remote_mgr",
        "//machina/c:c_api_internal",
        "//machina/c:tf_tensor_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime/eager:kernel_and_device",
        "//machina/core/common_runtime/eager:tensor_handle",
        "//machina/core/distributed_runtime:session_mgr",
        "//machina/core/distributed_runtime:test_utils",
        "//machina/core/distributed_runtime:worker_env",
        "//machina/core/distributed_runtime/rpc:rpc_rendezvous_mgr",
        "//machina/core/protobuf:eager_service_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
    ],
)

cc_library(
    name = "remote_mgr",
    srcs = [
        "remote_mgr.cc",
    ],
    hdrs = [
        "remote_mgr.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//machina:internal"],
    deps = [
        ":remote_tensor_handle",
        "//machina/core:lib",
        "//machina/core/common_runtime/eager:eager_executor",
        "//machina/core/common_runtime/eager:tensor_handle",
        "//machina/core/platform:error_payloads",
        "//machina/core/util:env_var",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "remote_mgr_test",
    size = "small",
    srcs = ["remote_mgr_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":remote_mgr",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime/eager:core",
        "//machina/core/common_runtime/eager:tensor_handle",
        "//machina/core/platform:error_payloads",
    ],
)

cc_library(
    name = "remote_tensor_handle_data",
    srcs = ["remote_tensor_handle_data.cc"],
    hdrs = ["remote_tensor_handle_data.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":destroy_tensor_handle_node",
        ":eager_client",
        "//machina/core:lib",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/profiler/lib:traceme",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "remote_copy_node",
    srcs = [
        "remote_copy_node.cc",
    ],
    hdrs = [
        "remote_copy_node.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//machina:internal"],
    deps = [
        ":remote_mgr",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/common_runtime/eager:attr_builder",
        "//machina/core/common_runtime/eager:eager_executor",
        "//machina/core/common_runtime/eager:eager_operation",
        "//machina/core/common_runtime/eager:tensor_handle",
        "@com_google_absl//absl/types:optional",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "eager_client.h",
        "remote_tensor_handle.h",
        "remote_tensor_handle_data.h",
    ],
    visibility = [
        "//machina/core/function/runtime_client:__pkg__",
        "//machina/python:__subpackages__",
    ],
)
