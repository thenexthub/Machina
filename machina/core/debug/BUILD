# Description:
# TensorFlow Debugger (tfdbg).
#
# Public target(s):
#
# ":debug" - Depending on this target causes a concrete implementation of
#    DebuggerState to be constructed at initialization time, enabling
#    TensorFlow Debugger (tfdbg) support. For details, please see
#    core/common_runtime/debugger_state_interface.h.
# ":debug_callback_registry" - Depending on this target exposes a global
#    callback registry that will be used to record any observed tensors matching
#    a watch state.
# ":debug_node_key" - Defines a struct used for tracking tensors.

# Placeholder: load py_proto_library
load(
    "//machina:machina.bzl",
    "check_deps",
    "if_windows",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_library",
)
load("//machina:machina.default.bzl", "tf_grpc_cc_dependencies")

# For platform specific build config
load(
    "//machina/core/platform:build_config.bzl",
    "tf_additional_all_protos",
    "tf_proto_library",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

# Check that machina/core:machina does not depend on grpc.
check_deps(
    name = "core_machina_check_deps",
    disallowed_deps = ["@com_github_grpc_grpc//:grpc++"],
    deps = ["//machina/core:machina"],
)

tf_proto_library(
    name = "debug_service_proto",
    srcs = [
        "debug_service.proto",
    ],
    has_services = 1,
    create_grpc_library = True,
    protodeps = [
        ":debugger_event_metadata_proto",
        "//machina/core/profiler:protos_all",
    ] + tf_additional_all_protos(),
    visibility = ["//machina:__subpackages__"],
)

tf_proto_library(
    name = "debugger_event_metadata_proto",
    srcs = ["debugger_event_metadata.proto"],
)

cc_library(
    name = "debug",
    srcs = ["debug.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":debugger_state_impl",
        "//machina/core:core_cpu_internal",
        "//machina/core:debug_ops_op_lib",
        # Depends on grpc and hence should stay out of
        # //third_party/machina/core.
        "//machina/core/kernels:debug_ops",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "debugger_state_impl",
    srcs = ["debugger_state_impl.cc"],
    hdrs = ["debugger_state_impl.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        ":debug_graph_utils",
        ":debug_io_utils",
        "//machina/core:core_cpu_internal",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "debug_graph_utils",
    srcs = ["debug_graph_utils.cc"],
    hdrs = ["debug_graph_utils.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

tf_cuda_library(
    name = "debug_io_utils",
    srcs = ["debug_io_utils.cc"],
    hdrs = ["debug_io_utils.h"],
    copts = tf_copts(),
    linkopts = if_windows(["-DEFAULTLIB:ws2_32.lib"]),
    linkstatic = 1,
    deps = [
        ":debug_callback_registry",
        ":debug_node_key",
        ":debug_service_cc_grpc_proto",
        ":debugger_event_metadata_proto_cc",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ] + tf_grpc_cc_dependencies(),
    alwayslink = 1,
)

tf_cuda_library(
    name = "debug_grpc_testlib",
    srcs = ["debug_grpc_testlib.cc"],
    hdrs = ["debug_grpc_testlib.h"],
    copts = tf_copts(),
    linkstatic = 1,
    deps = [
        ":debug_graph_utils",
        ":debug_io_utils",
        ":debug_service_cc_grpc_proto",
        ":debugger_event_metadata_proto_cc",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
    ] + tf_grpc_cc_dependencies(),
    alwayslink = 1,
)

tf_cuda_library(
    name = "debug_node_key",
    srcs = ["debug_node_key.cc"],
    hdrs = ["debug_node_key.h"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
    ],
)

tf_cc_test(
    name = "debug_io_utils_test",
    size = "small",
    srcs = ["debug_io_utils_test.cc"],
    tags = [
        "no_oss",  # TODO(b/137652456): remove when fixed
    ],
    deps = [
        ":debug_callback_registry",
        ":debug_grpc_testlib",
        ":debug_io_utils",
        ":debug_node_key",
        ":debugger_event_metadata_proto_cc",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

tf_cc_test(
    name = "debug_graph_utils_test",
    size = "small",
    srcs = ["debug_graph_utils_test.cc"],
    deps = [
        ":debug_graph_utils",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

tf_cc_test(
    name = "grpc_session_debug_test",
    size = "medium",
    srcs = ["grpc_session_debug_test.cc"],
    tags = [
        "no_oss",  # b/62956105: port conflicts.
        "nomac",  # b/38276817
    ],
    deps = [
        ":debug_grpc_testlib",
        ":debug_io_utils",
        "//machina/core:array_ops_op_lib",
        "//machina/core:core_cpu",
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
        "//machina/core:testlib",
        "//machina/core/distributed_runtime/rpc:grpc_server_lib",
        "//machina/core/distributed_runtime/rpc:grpc_session",
        "//machina/core/distributed_runtime/rpc:grpc_testlib",
        "//machina/core/kernels:constant_op",
        "//machina/core/kernels:matmul_op",
        "//machina/core/protobuf:master_proto_cc",
    ],
)

tf_cc_test(
    name = "debug_grpc_io_utils_test",
    size = "small",
    srcs = ["debug_grpc_io_utils_test.cc"],
    tags = [
        "no_oss",  # b/73962011
    ],
    deps = [
        ":debug_graph_utils",
        ":debug_grpc_testlib",
        ":debug_io_utils",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "debug_callback_registry",
    srcs = ["debug_callback_registry.cc"],
    hdrs = ["debug_callback_registry.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":debug_node_key",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ],
)

tf_cc_binary(
    name = "bfc_dump_reader",
    srcs = ["bfc_dump_reader.cc"],
    deps = [
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:regexp",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "debug_service_py_pb2",
#     has_services = 1,
#     deps = [":debug_service_proto"],
# )
#
# py_proto_library(
#     name = "debugger_event_metadata_py_pb2",
#     deps = [":debugger_event_metadata_proto"],
# )
# copybara:uncomment_end

# TODO(cais): Add the following back in when tfdbg is supported on Android.
# filegroup(
#     name = "android_srcs",
#     srcs = [
#         "debug_graph_utils.cc",
#         "debug_graph_utils.h",
#         "debug_io_utils.cc",
#         "debug_io_utils.h",
#     ],
#     visibility = ["//visibility:public"],
# )
