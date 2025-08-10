# Description:
# Profiler C API

load("//machina:machina.default.bzl", "filegroup")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "headers",
    srcs = [
        "pluggable_profiler.h",
    ],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "pluggable_profiler_hdrs",
    hdrs = ["pluggable_profiler.h"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/c:c_api_macros",
        "//machina/c:tf_status_headers",
    ],
)

cc_library(
    name = "pluggable_profiler",
    srcs = ["pluggable_profiler.cc"],
    hdrs = ["pluggable_profiler.h"],
    visibility = ["//machina:internal"],
    deps = [
        ":pluggable_profiler_internal",
        "//machina/c:c_api_macros",
        "//machina/c:tf_status",
        "//machina/c:tf_status_helper",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime/device:device_utils",
        "//machina/core/profiler/lib:profiler_factory",
        "//machina/core/profiler/lib:profiler_interface",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "pluggable_profiler_internal",
    hdrs = [
        "pluggable_profiler.h",
        "pluggable_profiler_internal.h",
    ],
    visibility = ["//machina/core/common_runtime/pluggable_device:__subpackages__"],
    deps = [
        "//machina/c:c_api_macros",
        "//machina/c:tf_status",
        "//machina/c:tf_status_helper",
        "//machina/core/platform:status",
        "//machina/core/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)
