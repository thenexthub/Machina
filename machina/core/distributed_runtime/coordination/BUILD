load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load("//machina:machina.default.bzl", "filegroup")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "coordination_client",
    hdrs = ["coordination_client.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_client",
    ],
)

cc_library(
    name = "coordination_service_rpc_handler",
    hdrs = [
        "coordination_service_rpc_handler.h",
    ],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service_rpc_handler",
    ],
)

cc_library(
    name = "coordination_service_error_util",
    hdrs = ["coordination_service_error_util.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service_error_util",
    ],
)

cc_library(
    name = "coordination_service_barrier_proxy",
    srcs = ["coordination_service_barrier_proxy.cc"],
    hdrs = ["coordination_service_barrier_proxy.h"],
    # copybara:uncomment copts = ["-Wthread-safety-analysis"],
    deps = [
        "//machina/core:framework_lite",
        "//machina/core/platform:macros",
        "//machina/core/platform:status",
        "//machina/core/platform:thread_annotations",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
        "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc",
    ],
)

tf_cc_test(
    name = "coordination_service_barrier_proxy_test",
    srcs = ["coordination_service_barrier_proxy_test.cc"],
    # copybara:uncomment extra_copts = ["-Wthread-safety-analysis"],
    deps = [
        ":coordination_service_barrier_proxy",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/distributed_runtime:call_options",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_client",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
        "@local_xla//xla/tsl/protobuf:coordination_config_proto_cc",
        "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "coordination_client.h",
    ],
    visibility = [
        "//machina/core/function/runtime_client:__pkg__",
        "//machina/python:__subpackages__",
    ],
)
