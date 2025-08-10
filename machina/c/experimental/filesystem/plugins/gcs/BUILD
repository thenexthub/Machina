# Experimental gcs filesystem plugin.
load("//machina:machina.bzl", "get_win_copts", "tf_cc_shared_object", "tf_cc_test")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

# Filesystem implementation for GCS environments
tf_cc_shared_object(
    name = "gcs_filesystem",
    framework_so = [],
    linkstatic = False,
    per_os_targets = 1,
    visibility = ["//visibility:public"],
    deps = [":gcs_filesystem_impl"],
)

# The real implementation of the filesystem.
cc_library(
    name = "gcs_filesystem_impl",
    srcs = ["gcs_filesystem.cc"],
    hdrs = ["gcs_filesystem.h"],
    copts = select({
        "//conditions:default": [],
        "//machina:windows": get_win_copts(),
    }),
    deps = [
        ":expiring_lru_cache",
        ":gcs_helper",
        ":ram_file_block_cache",
        "//machina/c:env",
        "//machina/c:logging",
        "//machina/c:tf_status",
        "//machina/c/experimental/filesystem:filesystem_interface",
        "@com_github_googlecloudplatform_google_cloud_cpp//:storage_client",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "gcs_helper",
    srcs = ["gcs_helper.cc"],
    hdrs = ["gcs_helper.h"],
    linkstatic = 1,
    deps = [
        "//machina/c:env",
    ],
)

cc_library(
    name = "cleanup",
    hdrs = ["cleanup.h"],
)

cc_library(
    name = "ram_file_block_cache",
    srcs = ["ram_file_block_cache.cc"],
    hdrs = ["ram_file_block_cache.h"],
    deps = [
        ":cleanup",
        "//machina/c:env",
        "//machina/c:logging",
        "//machina/c:tf_status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "ram_file_block_cache_test",
    size = "small",
    srcs = ["ram_file_block_cache_test.cc"],
    deps = [
        ":ram_file_block_cache",
        "//machina/c:tf_status_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform/cloud:now_seconds_env",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_test(
    name = "gcs_filesystem_test",
    srcs = [
        "gcs_filesystem_test.cc",
    ],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":gcs_filesystem_impl",
        "//machina/c:tf_status_helper",
        "//machina/core/platform:path",
        "//machina/core/platform:stacktrace_handler",
        "//machina/core/platform:test",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "expiring_lru_cache",
    hdrs = ["expiring_lru_cache.h"],
    deps = [
        "//machina/c:env",
        "//machina/c:tf_status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "expiring_lru_cache_test",
    size = "small",
    srcs = ["expiring_lru_cache_test.cc"],
    deps = [
        ":expiring_lru_cache",
        "//machina/c:tf_status_helper",
        "//machina/c:tf_status_internal",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform/cloud:now_seconds_env",
    ],
)
