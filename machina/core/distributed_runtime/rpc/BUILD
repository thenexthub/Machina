# Description:
#   RPC communication interfaces and implementations for TensorFlow.

load(
    "//machina:machina.bzl",
    "if_google",
    "if_windows",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_cuda_library",
)
load("//machina:machina.default.bzl", "filegroup", "tf_cuda_cc_test", "tf_cuda_cc_tests", "tf_grpc_cc_dependencies", "tf_grpc_dependencies")

# For platform specific build config
load(
    "//machina/core/platform:build_config.bzl",
    "tf_protos_profiler_service",
)
load(
    "//machina/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = if_google(
        ["//machina:internal"],
        ["//visibility:public"],
    ),
    licenses = ["notice"],
)

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "grpc_util",
    srcs = ["grpc_util.cc"],
    hdrs = ["grpc_util.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    linkopts = if_windows(["-DEFAULTLIB:ws2_32.lib"]),
    deps = [
        "//machina/core/distributed_runtime:error_payloads",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/core:lib",
        # Required to be able to overload TensorResponse parsing.
        "//machina/core/distributed_runtime:tensor_coding",
        "//machina/core:lib_internal",
        "@local_xla//xla/tsl/distributed_runtime/rpc:grpc_util",
    ] + tf_grpc_dependencies() + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_client_cq_tag",
    srcs = [],
    hdrs = ["grpc_client_cq_tag.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "@local_xla//xla/tsl/distributed_runtime/rpc:grpc_client_cq_tag",
    ],
)

cc_library(
    name = "grpc_state",
    srcs = ["grpc_state.cc"],
    hdrs = ["grpc_state.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_client_cq_tag",
        ":grpc_util",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/distributed_runtime:call_options",
        "//machina/core/distributed_runtime:tensor_coding",
        "@com_google_absl//absl/strings:str_format",
        "@local_xla//xla/tsl/distributed_runtime/rpc:grpc_state",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_remote_worker",
    srcs = ["grpc_remote_worker.cc"],
    hdrs = ["grpc_remote_worker.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_client_cq_tag",
        ":grpc_state",
        ":grpc_util",
        ":grpc_worker_service_impl",
        "//machina/core:core_cpu_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/distributed_runtime:call_options",
        "//machina/core/distributed_runtime:tensor_coding",
        "//machina/core/distributed_runtime:worker_cache_logger",
        "//machina/core/distributed_runtime:worker_interface",
        "//machina/core/protobuf:worker_proto_cc",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_channel",
    hdrs = ["grpc_channel.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "@local_xla//xla/tsl/distributed_runtime/rpc:grpc_channel",
    ],
)

cc_library(
    name = "grpc_tensor_coding",
    srcs = ["grpc_tensor_coding.cc"],
    hdrs = ["grpc_tensor_coding.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/protobuf:worker_proto_cc",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_worker_cache",
    srcs = ["grpc_worker_cache.cc"],
    hdrs = ["grpc_worker_cache.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_channel",
        ":grpc_client_cq_tag",
        ":grpc_remote_worker",
        ":grpc_util",
        "//machina/core:lib",
        "//machina/core/distributed_runtime:worker_cache",
        "//machina/core/distributed_runtime:worker_cache_logger",
        "//machina/core/distributed_runtime:worker_cache_partial",
        "//machina/core/distributed_runtime:worker_interface",
        "//machina/core/distributed_runtime/rpc/coordination:grpc_coordination_client",
        "//machina/core/distributed_runtime/rpc/eager:grpc_eager_client",
        "//machina/core/util:env_var",
    ],
)

tf_cc_test(
    name = "grpc_worker_cache_test",
    size = "small",
    srcs = [
        "grpc_worker_cache_test.cc",
    ],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_worker_cache",
        "//machina/c:tf_status_headers",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/distributed_runtime:test_utils",
        "//machina/core/distributed_runtime/rpc:grpc_channel",
        "//machina/core/platform:env",
        "//machina/core/platform:status",
        "//machina/core/platform:strcat",
    ],
)

cc_library(
    name = "rpc_response_cache",
    srcs = ["rpc_response_cache.cc"],
    hdrs = ["rpc_response_cache.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cuda_library(
    name = "grpc_worker_service",
    srcs = ["grpc_worker_service.cc"],
    hdrs = ["grpc_worker_service.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_tensor_coding",
        ":grpc_util",
        ":grpc_worker_service_impl",
        ":rpc_response_cache",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/distributed_runtime:graph_mgr",
        "//machina/core/distributed_runtime:rendezvous_mgr_interface",
        "//machina/core/distributed_runtime:worker",
        "//machina/core/distributed_runtime:worker_cache",
        "//machina/core/distributed_runtime:worker_env",
        "//machina/core/distributed_runtime:worker_session",
        "//machina/core/profiler/lib:scoped_memory_debug_annotation",
        "//machina/core/protobuf:worker_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/distributed_runtime/rpc:async_service_interface",
        "@local_xla//xla/tsl/distributed_runtime/rpc:grpc_call",
        "@local_xla//xla/tsl/protobuf:rpc_options_proto_cc",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_worker_service_impl",
    srcs = ["grpc_worker_service_impl.cc"],
    hdrs = ["grpc_worker_service_impl.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_util",
        "//machina/core/distributed_runtime:tensor_coding",
        "//machina/core/protobuf:worker_proto_cc",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_remote_master",
    srcs = ["grpc_remote_master.cc"],
    hdrs = ["grpc_remote_master.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_master_service_impl",
        ":grpc_util",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/distributed_runtime:call_options",
        "//machina/core/distributed_runtime:master_interface",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/protobuf:master_proto_cc",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:retrying_utils",
        "@local_xla//xla/tsl/platform:env",
    ],
    alwayslink = 1,
)

cc_library(
    name = "grpc_master_service",
    srcs = ["grpc_master_service.cc"],
    hdrs = ["grpc_master_service.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_master_service_impl",
        ":grpc_util",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/distributed_runtime:master",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/protobuf:master_proto_cc",
        "@local_xla//xla/tsl/distributed_runtime/rpc:async_service_interface",
        "@local_xla//xla/tsl/distributed_runtime/rpc:grpc_call",
    ] + tf_grpc_cc_dependencies(),
    alwayslink = 1,
)

cc_library(
    name = "grpc_master_service_impl",
    srcs = ["grpc_master_service_impl.cc"],
    hdrs = ["grpc_master_service_impl.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core/protobuf:master_proto_cc",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "rpc_rendezvous_mgr",
    srcs = ["rpc_rendezvous_mgr.cc"],
    hdrs = ["rpc_rendezvous_mgr.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/distributed_runtime:base_rendezvous_mgr",
        "//machina/core/distributed_runtime:request_id",
        "//machina/core/distributed_runtime:tensor_coding",
        "//machina/core/distributed_runtime:worker_cache",
        "//machina/core/distributed_runtime:worker_env",
        "//machina/core/distributed_runtime:worker_interface",
    ],
)

cc_library(
    name = "grpc_server_lib",
    srcs = ["grpc_server_lib.cc"],
    hdrs = ["grpc_server_lib.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    deps = [
        ":grpc_channel",
        ":grpc_master_service",
        ":grpc_worker_cache",
        ":grpc_worker_service",
        ":rpc_rendezvous_mgr",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/distributed_runtime:collective_param_resolver_distributed",
        "//machina/core/distributed_runtime:device_resolver_distributed",
        "//machina/core/distributed_runtime:graph_mgr",
        "//machina/core/distributed_runtime:local_master",
        "//machina/core/distributed_runtime:master",
        "//machina/core/distributed_runtime:master_env",
        "//machina/core/distributed_runtime:master_session",
        "//machina/core/distributed_runtime:rpc_collective_executor_mgr",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/distributed_runtime:session_mgr",
        "//machina/core/distributed_runtime:worker_cache_wrapper",
        "//machina/core/distributed_runtime:worker_env",
        "//machina/core/distributed_runtime/rpc/coordination:grpc_coordination_service_impl",
        "//machina/core/distributed_runtime/rpc/eager:grpc_eager_service_impl",
        "//machina/core/nccl:collective_communicator",
        "//machina/core/profiler/rpc:profiler_service_impl",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service",
        "@local_xla//xla/tsl/distributed_runtime/rpc:async_service_interface",
    ] + tf_protos_profiler_service() + tf_grpc_dependencies() + tf_grpc_cc_dependencies(),
    alwayslink = 1,
)

cc_library(
    name = "grpc_runtime",
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    visibility = ["//visibility:public"],
    deps = [
        ":grpc_server_lib",
        ":grpc_session",
    ],
)

tf_cc_binary(
    name = "grpc_machina_server",
    srcs = [
        "grpc_machina_server.cc",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_server_lib",
        "//machina/core:core_cpu",
        "//machina/core:data_flow_ops_op_lib",
        "//machina/core:framework_internal",
        "//machina/core:functional_ops_op_lib",
        "//machina/core:lib",
        "//machina/core:lookup_ops_op_lib",
        "//machina/core:math_ops_op_lib",
        "//machina/core:no_op_op_lib",
        "//machina/core:protos_all_cc",
        "//machina/core:sendrecv_ops_op_lib",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/kernels:data_flow",
    ] + tf_grpc_cc_dependencies(),
)

# Library version of grpc_testlib_server, to allow for custom testlib servers.
cc_library(
    name = "grpc_testlib_server_main",
    testonly = 1,
    srcs = [
        "grpc_testlib_server.cc",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_server_lib",
        "//machina/core:array_ops_op_lib",
        "//machina/core:bitwise_ops_op_lib",
        "//machina/core:core_cpu",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core:protos_all_cc",
        "//machina/core:state_ops_op_lib",
        "//machina/core:testlib",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/kernels:constant_op",
        "//machina/core/kernels:cwise_op",
        "//machina/core/kernels:dense_update_ops",
        "//machina/core/kernels:identity_op",
        "//machina/core/kernels:matmul_op",
        "//machina/core/kernels:reduction_ops",
        "//machina/core/kernels:variable_ops",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
    ] + tf_grpc_cc_dependencies(),
    alwayslink = 1,
)

tf_cc_binary(
    name = "grpc_testlib_server",
    testonly = 1,
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [":grpc_testlib_server_main"],
)

tf_cuda_library(
    name = "grpc_testlib",
    testonly = 1,
    srcs = ["grpc_testlib.cc"],
    hdrs = ["grpc_testlib.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    data = [
        ":grpc_testlib_server",
    ],
    visibility = ["//machina:internal"],
    deps = [
        ":grpc_session",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:testlib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
    alwayslink = 1,
)

cc_library(
    name = "grpc_session",
    srcs = ["grpc_session.cc"],
    hdrs = ["grpc_session.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_channel",
        ":grpc_remote_master",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/distributed_runtime:call_options",
        "//machina/core/distributed_runtime:local_master",
        "//machina/core/distributed_runtime:master_interface",
        "//machina/core/distributed_runtime:message_wrappers",
        "//machina/core/distributed_runtime:request_id",
        "//machina/core/protobuf:master_proto_cc",
    ],
    alwayslink = 1,
)

tf_cuda_cc_tests(
    name = "rpc_tests",
    size = "small",
    srcs = [
        "rpc_rendezvous_mgr_test.cc",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    linkopts = select({
        "//machina:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    tags = tf_cuda_tests_tags() + [],
    deps = [
        ":grpc_channel",
        ":grpc_server_lib",
        ":grpc_session",
        ":grpc_testlib",
        ":rpc_rendezvous_mgr",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/distributed_runtime:test_utils",
        "//machina/core/platform:blocking_counter",
        "//machina/core/protobuf:master_proto_cc",
    ],
)

tf_cc_test(
    name = "grpc_tensor_coding_test",
    size = "small",
    srcs = ["grpc_tensor_coding_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = [
        "no_mac",
        "no_windows",
    ],
    deps = [
        ":grpc_tensor_coding",
        ":grpc_testlib",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/protobuf:worker_proto_cc",
        "@com_google_absl//absl/status",
    ] + tf_grpc_cc_dependencies(),
)

tf_cuda_cc_test(
    name = "grpc_session_test",
    size = "medium",
    # highly variable test time. Typically ~2m, but can spike to 5+min.
    timeout = "long",
    srcs = ["grpc_session_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    tags = tf_cuda_tests_tags() + [
        "no_cuda_asan",  # b/176448181
        "no_oss",  # b/62956105: port conflicts.
    ],
    deps = [
        ":grpc_channel",
        ":grpc_server_lib",
        ":grpc_session",
        ":grpc_testlib",
        ":rpc_rendezvous_mgr",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:state_ops_op_lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/kernels:constant_op",
        "//machina/core/kernels:dense_update_ops",
        "//machina/core/kernels:matmul_op",
        "//machina/core/kernels:variable_ops",
        "//machina/core/protobuf:master_proto_cc",
    ],
)
