load("@rules_cc//cc:cc_test.bzl", "cc_test")

# Description:
#
# This package contains shim library targets for the Async C package.
load("//machina/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:private",
    ],
    licenses = ["notice"],
)

cc_library_with_tflite(
    name = "backend_async_kernel_interface",
    srcs = ["backend_async_kernel_interface.cc"],
    hdrs = ["backend_async_kernel_interface.h"],
    tflite_deps = [
        "//machina/lite/async/c:async_kernel",
        "//machina/lite/async/c:types",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "backend_async_kernel_interface_test",
    srcs = ["backend_async_kernel_interface_test.cc"],
    deps = [
        ":backend_async_kernel_interface",
        "//machina/lite/async/c:types",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core/async:async_kernel_internal",
        "//machina/lite/core/async/testing:mock_async_kernel",
        "@com_google_googletest//:gtest_main",
    ],
)
