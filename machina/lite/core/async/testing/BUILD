load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Test utilities for TFLite async execution.
load("//machina/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "test_backend",
    testonly = 1,
    srcs = ["test_backend.cc"],
    hdrs = ["test_backend.h"],
    deps = [
        "//machina/lite:array",
        "//machina/lite:builtin_ops",
        "//machina/lite/core/async:async_kernel_internal",
        "//machina/lite/core/async/c:types",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/delegates:utils",
    ],
)

cc_library_with_tflite(
    name = "mock_async_kernel",
    testonly = 1,
    hdrs = ["mock_async_kernel.h"],
    deprecation = "Use //machina/lite/async/testing:mock_async_kernel instead.",
    tflite_deps = [
        "//machina/lite/async/testing:mock_async_kernel",
    ],
)
