load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/lite:build_def.bzl", "tflite_cc_library_with_c_headers_test")
load("//machina/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# Client of this library is responsible to provide an implementation of
# for headers in //third_party/machina/lite/nnapi:nnapi_implementation_headers
cc_library(
    name = "nnapi_delegate_no_nnapi_implementation",
    srcs = select({
        "//machina:ios": [
            "nnapi_delegate_disabled.cc",
        ],
        "//machina:windows": [
            "nnapi_delegate_disabled.cc",
        ],
        "//conditions:default": [
            "nnapi_delegate.cc",
            "nnapi_delegate_c_api.cc",
            "quant_lstm_sup.cc",
            "quant_lstm_sup.h",
        ],
    }),
    hdrs = [
        "nnapi_delegate.h",
        "nnapi_delegate_c_api.h",
        "nnapi_delegate_kernel.h",
        "nnapi_delegate_plugin.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/lite:allocation",
        "//machina/lite:array",
        "//machina/lite:kernel_api",
        "//machina/lite:minimal_logging",
        "//machina/lite:util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/delegates:serialization",
        "//machina/lite/delegates:utils",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels/internal/utils:sparsity_format_converter",
        "//machina/lite/nnapi:nnapi_implementation_headers",
        "//machina/lite/nnapi:nnapi_lib",
        "//machina/lite/nnapi:nnapi_util",
        "//machina/lite/nnapi/sl:nnapi_support_library_headers",
        "//machina/lite/schema:schema_fbs",
        "@FP16",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "nnapi_delegate",
    hdrs = [
        "nnapi_delegate.h",
        "nnapi_delegate_c_api.h",
        "nnapi_delegate_kernel.h",
        "nnapi_delegate_plugin.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":nnapi_delegate_no_nnapi_implementation",
        "//machina/lite:allocation",
        "//machina/lite/core/c:common",
        "//machina/lite/delegates:serialization",
        "//machina/lite/nnapi:nnapi_implementation",
        "//machina/lite/nnapi:nnapi_lib",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "nnapi_delegate_c_api",
    hdrs = ["nnapi_delegate_c_api.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],
    deps = [
        ":nnapi_delegate",
        "//machina/lite/core/c:common",
    ],
)

exports_files(["nnapi_delegate_c_api.h"])

cc_library(
    name = "nnapi_delegate_verbose_validation",
    srcs = select({
        "//machina:ios": [
            "nnapi_delegate_disabled.cc",
        ],
        "//machina:windows": [
            "nnapi_delegate_disabled.cc",
        ],
        "//conditions:default": [
            "nnapi_delegate.cc",
            "quant_lstm_sup.cc",
            "quant_lstm_sup.h",
        ],
    }),
    hdrs = [
        "nnapi_delegate.h",
        "nnapi_delegate_kernel.h",
        "nnapi_delegate_plugin.h",
    ],
    copts = ["-DNNAPI_VERBOSE_VALIDATION"],
    deps = [
        "//machina/lite:allocation",
        "//machina/lite:array",
        "//machina/lite:kernel_api",
        "//machina/lite:minimal_logging",
        "//machina/lite:util",
        "//machina/lite/c:common",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/delegates:serialization",
        "//machina/lite/delegates:utils",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels/internal/utils:sparsity_format_converter",
        "//machina/lite/nnapi:nnapi_implementation",
        "//machina/lite/nnapi:nnapi_lib",
        "//machina/lite/nnapi:nnapi_util",
        "//machina/lite/nnapi/sl:nnapi_support_library_headers",
        "//machina/lite/schema:schema_fbs",
        "@FP16",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "acceleration_test_util",
    testonly = 1,
    srcs = [
        "acceleration_test_list.cc",
        "acceleration_test_util.cc",
    ],
    hdrs = ["acceleration_test_util.h"],
    deps = [
        ":nnapi_delegate",
        "//machina/lite/kernels:acceleration_test_util_internal",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "nnapi_delegate_mock_test",
    testonly = 1,
    hdrs = [
        "nnapi_delegate_mock_test.h",
    ],
    copts = select({
        "//machina:ios": [
            "-DNNAPI_DELEGATE_DISABLED",
        ],
        "//machina:windows": [
            "-DNNAPI_DELEGATE_DISABLED",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":nnapi_delegate",
        "//machina/lite/nnapi:nnapi_handler",
        "//machina/lite/nnapi:nnapi_implementation",
        "//machina/lite/nnapi:nnapi_lib",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "nnapi_delegate_test",
    size = "small",
    srcs = [
        "nnapi_delegate_test.cc",
    ],
    tags = [
        "no_windows",
        # TODO(b/401046768): Re-enable once the test is fixed.
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        "//machina/lite:framework",
        "//machina/lite:string",
        "//machina/lite:string_util",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/kernels:deprecated_backends",
        "//machina/lite/kernels:test_util",
        "//machina/lite/nnapi:nnapi_implementation",
        "//machina/lite/nnapi:nnapi_lib",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/log:check",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_c_api_test",
    size = "small",
    srcs = [
        "nnapi_delegate_c_api_test.cc",
    ],
    tags = [
        "no_windows",
        # TODO(b/401046768): Re-enable once the test is fixed.
        "notap",
        "tflite_not_portable_ios",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":nnapi_delegate",
        "//machina/lite:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/kernels:deprecated_backends",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_errno_test",
    size = "small",
    srcs = [
        "nnapi_delegate_errno_test.cc",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        ":nnapi_delegate_mock_test",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "//machina/lite/nnapi:nnapi_implementation",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_device_selection_test",
    size = "small",
    srcs = [
        "nnapi_delegate_device_selection_test.cc",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        ":nnapi_delegate_mock_test",
        "//machina/lite:framework",
        "//machina/lite:kernel_api",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "//machina/lite/nnapi:nnapi_implementation",
        "//machina/lite/nnapi:nnapi_lib",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_nnapi_failure_handling_test",
    size = "small",
    srcs = [
        "nnapi_delegate_nnapi_failure_handling_test.cc",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        ":nnapi_delegate_mock_test",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "//machina/lite/nnapi:nnapi_implementation",
        "//machina/lite/nnapi:nnapi_lib",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_signed_quantization_test",
    size = "small",
    srcs = [
        "nnapi_delegate_signed_quantization_test.cc",
    ],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":nnapi_delegate",
        ":nnapi_delegate_mock_test",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/kernels:test_util",
        "//machina/lite/nnapi:nnapi_implementation",
        "//machina/lite/nnapi:nnapi_lib",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "quant_lstm_sup_test",
    size = "small",
    srcs = [
        "quant_lstm_sup.cc",
        "quant_lstm_sup.h",
        "quant_lstm_sup_test.cc",
    ],
    deps = [
        "//machina/lite:framework",
        "//machina/lite/c:common",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
