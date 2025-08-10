load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "async_kernel_internal",
    hdrs = ["async_kernel_internal.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/lite/core/async/c:types",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
    ],
)

cc_library(
    name = "task_internal",
    srcs = ["task_internal.cc"],
    hdrs = ["task_internal.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":async_kernel_internal",
        "//machina/lite/core/async/c:types",
        "//machina/lite/core/async/interop/c:types",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
    ],
)

cc_test(
    name = "task_internal_test",
    srcs = ["task_internal_test.cc"],
    deps = [
        ":task_internal",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core/async/c:types",
        "//machina/lite/core/async/interop/c:types",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "async_subgraph",
    srcs = ["async_subgraph.cc"],
    hdrs = ["async_subgraph.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":async_kernel_internal",
        ":task_internal",
        "//machina/lite:minimal_logging",
        "//machina/lite/core:subgraph",
        "//machina/lite/core/async/c:types",
        "//machina/lite/core/async/interop/c:types",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
    ],
)

cc_test(
    name = "async_subgraph_test",
    srcs = ["async_subgraph_test.cc"],
    deps = [
        ":async_kernel_internal",
        ":async_subgraph",
        ":backend_async_kernel_interface",
        ":task_internal",
        "//machina/lite:framework",
        "//machina/lite/core:framework_stable",
        "//machina/lite/core/async/c:types",
        "//machina/lite/core/async/interop:attribute_map_internal",
        "//machina/lite/core/async/interop/c:types",
        "//machina/lite/core/async/testing:mock_async_kernel",
        "//machina/lite/core/async/testing:test_backend",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite(
    name = "backend_async_kernel_interface",
    hdrs = ["backend_async_kernel_interface.h"],
    deprecation = "Use //machina/lite/async:backend_async_kernel_interface instead.",
    tflite_deps = [
        "//machina/lite/async:backend_async_kernel_interface",
    ],
)

cc_library(
    name = "async_signature_runner",
    srcs = ["async_signature_runner.cc"],
    hdrs = ["async_signature_runner.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":async_kernel_internal",
        ":async_subgraph",
        ":task_internal",
        "//machina/lite/core:subgraph",
        "//machina/lite/core/async/c:types",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/internal:signature_def",
    ],
)

cc_test(
    name = "async_signature_runner_test",
    srcs = ["async_signature_runner_test.cc"],
    deps = [
        ":async_kernel_internal",
        ":async_signature_runner",
        ":backend_async_kernel_interface",
        "//machina/lite:framework",
        "//machina/lite:interpreter_test_util",
        "//machina/lite/core:framework_stable",
        "//machina/lite/core/async/c:task",
        "//machina/lite/core/async/c:types",
        "//machina/lite/core/async/interop:attribute_map_internal",
        "//machina/lite/core/async/testing:mock_async_kernel",
        "//machina/lite/core/async/testing:test_backend",
        "//machina/lite/core/c:c_api_experimental",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "@com_google_googletest//:gtest_main",
    ],
)
