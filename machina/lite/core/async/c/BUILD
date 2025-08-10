# This package contains the C API libraries for asynchronous execution and buffer interop.
# For clients using async APIs, please use machina/lite/async/c instead of this package.
# NOTE: Targets in this package are experimental.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load(
    "//machina/lite:build_def.bzl",
    "tflite_cc_library_with_c_headers_test",
    "tflite_copts",
    "tflite_copts_warnings",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "types.h",
    ],
    visibility = [
        "//machina/lite:__subpackages__",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "types",
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        # TODO(b/271281434): Update targets that depend on this to instead use
        # machina/lite/async/c.
        "//visibility:public",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "async_kernel",
    srcs = ["async_kernel.cc"],
    hdrs = ["async_kernel.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        # TODO(b/271281434): Update targets that depend on this to instead use
        # machina/lite/async/c.
        "//visibility:public",
    ],
    deps = [
        ":types",
        "//machina/lite/core/async:async_kernel_internal",
        "//machina/lite/core/async/interop/c:attribute_map",
        "//machina/lite/core/async/interop/c:types",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "task",
    srcs = ["task.cc"],
    hdrs = ["task.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        # TODO(b/271281434): Update targets that depend on this to instead use
        # machina/lite/async/c.
        "//visibility:public",
    ],
    deps = [
        ":types",
        "//machina/lite/core/async:task_internal",
        "//machina/lite/core/async/interop/c:types",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
    ],
)

cc_test(
    name = "task_test",
    srcs = ["task_test.cc"],
    deps = [
        ":task",
        ":types",
        "//machina/lite/core/async:task_internal",
        "//machina/lite/core/async/interop/c:types",
        "//machina/lite/core/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "async_signature_runner",
    srcs = ["async_signature_runner.cc"],
    hdrs = ["async_signature_runner.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = [
        # TODO(b/271281434): Update targets that depend on this to instead use
        # machina/lite/async/c.
        "//visibility:public",
    ],
    deps = [
        ":internal",
        ":types",
        "//machina/lite/c:c_api_internal",
        "//machina/lite/core/async:async_signature_runner",
        "//machina/lite/core/async/interop/c:attribute_map",
        "//machina/lite/core/async/interop/c:types",
        "//machina/lite/core/c:c_api",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
    ],
)

cc_test(
    name = "async_signature_runner_test",
    srcs = ["async_signature_runner_test.cc"],
    copts = tflite_copts() + tflite_copts_warnings(),
    data = [
        "//machina/lite:testdata/no_signatures.bin",
    ],
    deps = [
        ":async_signature_runner",
        ":internal",
        ":task",
        ":types",
        "//machina/lite:interpreter_test_util",
        "//machina/lite/c:c_api_experimental",
        "//machina/lite/c:c_api_internal",
        "//machina/lite/core:cc_api_stable",
        "//machina/lite/core/async:async_kernel_internal",
        "//machina/lite/core/async:backend_async_kernel_interface",
        "//machina/lite/core/async/interop/c:attribute_map",
        "//machina/lite/core/async/interop/c:types",
        "//machina/lite/core/async/testing:mock_async_kernel",
        "//machina/lite/core/async/testing:test_backend",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:c_api_without_op_resolver",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "internal",
    hdrs = ["internal.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = ["//visibility:private"],
    deps = ["//machina/lite/core/async:async_signature_runner"],
)
