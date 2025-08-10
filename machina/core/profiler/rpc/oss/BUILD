load("//machina:machina.default.bzl", "tf_grpc_cc_dependencies")
load("//machina/core/platform:rules_cc.bzl", "cc_library")
load("//machina/core/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/core/profiler/rpc:__pkg__",  # Scheuklappen: keep
    ],
    licenses = ["notice"],
)

cc_library(
    name = "grpc",
    srcs = [
        "grpc.cc",
        "//machina/core/profiler/rpc:grpc.h",
    ],
    copts = tf_profiler_copts(),
    deps = tf_grpc_cc_dependencies(),
    alwayslink = True,
)

exports_files(
    [
        "grpc.cc",
    ],
    visibility = ["//machina/core/profiler/rpc:__subpackages__"],
)
