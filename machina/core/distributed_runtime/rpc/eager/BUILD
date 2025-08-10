load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "tf_grpc_cc_dependencies")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "grpc_eager_service",
    srcs = ["grpc_eager_service.h"],
    hdrs = ["grpc_eager_service.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core/protobuf:eager_service_cc_grpc_proto",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_eager_client",
    srcs = ["grpc_eager_client.cc"],
    hdrs = ["grpc_eager_client.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_eager_service",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/distributed_runtime:call_options",
        "//machina/core/distributed_runtime/eager:eager_client",
        "//machina/core/distributed_runtime/rpc:grpc_channel",
        "//machina/core/distributed_runtime/rpc:grpc_client_cq_tag",
        "//machina/core/distributed_runtime/rpc:grpc_state",
        "//machina/core/distributed_runtime/rpc:grpc_util",
        "//machina/core/platform:error_payloads",
        "//machina/core/protobuf:eager_service_proto_cc",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/distributed_runtime:call_options",
    ] + tf_grpc_cc_dependencies(),
)

cc_library(
    name = "grpc_eager_service_impl",
    srcs = ["grpc_eager_service_impl.cc"],
    hdrs = ["grpc_eager_service_impl.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_eager_service",
        "//machina/core:framework",
        "//machina/core/distributed_runtime/eager:eager_service_impl",
        "//machina/core/distributed_runtime/rpc:grpc_channel",
        "//machina/core/distributed_runtime/rpc:grpc_util",
        "//machina/core/distributed_runtime/rpc:grpc_worker_cache",
        "//machina/core/protobuf:eager_service_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/distributed_runtime/rpc:async_service_interface",
        "@local_xla//xla/tsl/distributed_runtime/rpc:grpc_call",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "grpc_eager_client_test",
    size = "small",
    srcs = [
        "grpc_eager_client_test.cc",
    ],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":grpc_eager_client",
        "//machina/c:tf_status_headers",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/distributed_runtime/rpc:grpc_channel",
        "//machina/core/platform:blocking_counter",
        "//machina/core/platform:env",
        "//machina/core/platform:status",
        "//machina/core/platform:strcat",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)
