# copybara:uncomment(oss-unused) load("//net/grpc/go/build_defs:go_grpc_library.bzl", "go_grpc_library")
# Placeholder: load py_proto_library
load("//machina:machina.bzl", "tf_cc_binary")
load("//machina:machina.default.bzl", "filegroup")
load("//machina/core/platform:build_config.bzl", "tf_additional_all_protos", "tf_proto_library")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = ["//machina:internal"],
)

package_group(
    name = "internal",
    packages = [
        "//machina/core/profiler/...",
        "//machina/python/eager/...",
        "//machina/python/profiler/...",
        "//machina/python/tpu/profiler/...",
    ],
)

tf_proto_library(
    name = "profiler_service_monitor_result_proto",
    srcs = ["profiler_service_monitor_result.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "@local_tsl//tsl/profiler/protobuf:profiler_service_monitor_result_proto",
    ],
    visibility = ["//visibility:public"],
    exports = [
        "@local_tsl//tsl/profiler/protobuf:profiler_service_monitor_result_proto",
    ],
)

tf_proto_library(
    name = "profiler_options_proto",
    srcs = ["profiler_options.proto"],
    make_default_target_header_only = True,
    protodeps = ["@local_tsl//tsl/profiler/protobuf:profiler_options_proto"],
    visibility = ["//visibility:public"],
    exports = ["@local_tsl//tsl/profiler/protobuf:profiler_options_proto"],
)

# This is needed because of how tf_android_core_proto_sources parses proto paths.
exports_files(
    srcs = ["profiler_options.proto"],
    visibility = ["//machina/core:__pkg__"],
)

tf_proto_library(
    name = "profiler_service_proto",
    srcs = ["profiler_service.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "@local_tsl//tsl/profiler/protobuf:profiler_service_proto",
    ],
    use_grpc_namespace = True,
    visibility = ["//visibility:public"],
    exports = ["@local_tsl//tsl/profiler/protobuf:profiler_service_proto"],
)

tf_proto_library(
    name = "profiler_analysis_proto",
    srcs = ["profiler_analysis.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "@local_tsl//tsl/profiler/protobuf:profiler_analysis_proto",
    ],
    use_grpc_namespace = True,
    visibility = ["//visibility:public"],
    exports = ["@local_tsl//tsl/profiler/protobuf:profiler_analysis_proto"],
)

tf_proto_library(
    name = "protos_all",
    srcs = [
        "profile.proto",
        "tfprof_log.proto",
        "tfprof_options.proto",
        "tfprof_output.proto",
    ],
    protodeps = [
        "@local_tsl//tsl/profiler/protobuf:profile_proto",
    ] + tf_additional_all_protos(),
    visibility = [":friends"],
    exports = [
        "@local_tsl//tsl/profiler/protobuf:profile_proto",
    ],
)

cc_library(
    name = "profiler_impl",
    visibility = [
        "//machina:__pkg__",
        "//machina/python:__pkg__",
        "//machina/security/fuzzing/cc/ops:__pkg__",
    ],
    deps = [
        "//machina/core/profiler/lib:profiler_factory_impl",
        "//machina/core/profiler/lib:profiler_session_impl",
        "@local_xla//xla/tsl/profiler/backends/cpu:annotation_stack_impl",
        "@local_xla//xla/tsl/profiler/backends/cpu:threadpool_listener",
        "@local_xla//xla/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@local_xla//xla/tsl/profiler/utils:time_utils_impl",
    ],
    alwayslink = True,
)

# Libraries and binaries of TensorFlow profiler 1.x.
tf_cc_binary(
    name = "profiler",
    srcs = ["profiler.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":protos_all_cc",
        "//machina/c:checkpoint_reader",
        "//machina/c:tf_status",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_proto_parsing",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/platform:protobuf",
        "//machina/core/profiler:tfprof_options",
        "//machina/core/profiler/internal:tfprof_stats",
        "//machina/core/profiler/internal:tfprof_utils",
        "//machina/core/profiler/internal/advisor:tfprof_advisor",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@linenoise",
    ],
)

cc_library(
    name = "tfprof_options",
    srcs = ["tfprof_options.cc"],
    hdrs = ["tfprof_options.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":protos_all_cc",
        "//machina/core:framework_headers_lib",
        "//machina/core:framework_types_hdr",
        "//machina/core:portable_gif_internal",
        "//machina/core/lib/core:status",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "//machina/core/profiler/lib:mobile_srcs_no_runtime",
        "@local_tsl//tsl/profiler/lib:mobile_srcs_no_runtime",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "//machina/core/profiler/lib:mobile_srcs_only_runtime",
        "@local_tsl//tsl/profiler/lib:mobile_srcs_only_runtime",
    ],
    visibility = ["//machina/core:__pkg__"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "profiler_analysis_proto_py_pb2",
#     has_services = 1,
#     visibility = ["//visibility:public"],
#     deps = [":profiler_analysis_proto"],
# )
#
# py_proto_library(
#     name = "protos_all_py_pb2",
#     visibility = [":friends"],
#     deps = [":protos_all"],
# )
# copybara:uncomment_end
