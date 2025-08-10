load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Description:
# test for stream_executor
load(
    "//machina:machina.bzl",
    "tf_cc_shared_object",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_cc_shared_object(
    name = "test_pluggable_device.so",
    srcs = ["test_pluggable_device.cc"],
    visibility = ["//machina/c:__subpackages__"],
    deps = [
        "//machina/c/experimental/stream_executor:stream_executor_hdrs",
        "//machina/c/experimental/stream_executor:stream_executor_test_util",
    ],
)

cc_library(
    name = "test_pluggable_device",
    srcs = ["test_pluggable_device.cc"],
    visibility = ["//machina/core:__subpackages__"],
    deps = [
        "//machina/c/experimental/stream_executor:stream_executor_hdrs",
        "//machina/c/experimental/stream_executor:stream_executor_test_util",
    ],
)
