# Test utilities for TFLite async execution.
load("//machina/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:private",
    ],
    licenses = ["notice"],
)

cc_library_with_tflite(
    name = "mock_async_kernel",
    testonly = 1,
    hdrs = ["mock_async_kernel.h"],
    tflite_deps = [
        "//machina/lite/async:backend_async_kernel_interface",
        "//machina/lite/async/c:types",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_googletest//:gtest",
    ],
)
