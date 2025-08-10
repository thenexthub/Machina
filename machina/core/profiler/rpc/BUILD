load("//machina:machina.default.bzl", "tf_external_workspace_visible", "tf_grpc_cc_dependencies")
load("//machina/core/platform:rules_cc.bzl", "cc_library")
load(
    "//machina/core/profiler/builds:build_config.bzl",
    "tf_profiler_alias",
    "tf_profiler_copts",
    "tf_profiler_pybind_cc_library_wrapper",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/core/profiler:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "grpc",
    hdrs = ["grpc.h"],
    deps = [
        tf_profiler_alias("//machina/core/profiler/rpc/", "grpc"),
    ] + tf_grpc_cc_dependencies(),
)

exports_files(
    [
        "grpc.h",
    ],
    visibility = ["//machina/core/profiler/rpc:__subpackages__"],
)

# Linked to pywrap_machina.
cc_library(
    name = "profiler_service_impl",
    hdrs = ["profiler_service_impl.h"],
    copts = tf_profiler_copts(),
    visibility = tf_external_workspace_visible(
        [
            "//machina/core/data/service:__pkg__",
            "//machina/core/distributed_runtime/rpc:__pkg__",
            "//machina_serving/model_servers:__pkg__",
        ],
    ),
    deps = [
        "//machina/core:lib",
        "//machina/core/profiler/lib:profiler_session",
        "//machina/core/profiler/rpc/client:save_profile",
        "//machina/core/profiler/utils:math_utils",
        "//machina/core/profiler/utils:xplane_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "@local_tsl//tsl/profiler/protobuf:profiler_service_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/tsl/profiler/rpc:profiler_service_impl",
        "@local_xla//xla/tsl/profiler/utils:file_system_utils",
        "@local_xla//xla/tsl/profiler/utils:time_utils",
    ] + tf_grpc_cc_dependencies(),
)

tf_profiler_pybind_cc_library_wrapper(
    name = "profiler_server_for_pybind",
    actual = ":profiler_server_impl",
    visibility = [
        "//machina/python/profiler/internal:__pkg__",
        "@org_xprof//xprof/pywrap:__pkg__",
    ],
)

cc_library(
    name = "profiler_server_impl",
    hdrs = ["profiler_server.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//machina:__pkg__",
        "//machina/core/profiler:internal",
        "//machina/python:__pkg__",
        "//machina/python/profiler/internal:__pkg__",
        "@local_xla//xla:__subpackages__",
        "@local_xla//xla/tsl/profiler/rpc/client:__pkg__",
        "@org_xprof//xprof/pywrap:__pkg__",
    ],
    deps = [
        ":profiler_service_impl",
        "//machina/core:lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/protobuf:profiler_service_cc_grpc_proto",
        "@local_xla//xla/tsl/profiler/rpc:profiler_server_impl",
    ] + tf_grpc_cc_dependencies(),
    alwayslink = True,
)
