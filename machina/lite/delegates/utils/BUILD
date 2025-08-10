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
    name = "simple_delegate",
    srcs = [
        "simple_delegate.cc",
    ],
    hdrs = [
        "simple_delegate.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/lite:array",
        "//machina/lite:kernel_api",
        "//machina/lite:minimal_logging",
        "//machina/lite/core/c:common",
        "//machina/lite/delegates:utils",
        "//machina/lite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "simple_delegate_test",
    srcs = ["simple_delegate_test.cc"],
    deps = [
        "//machina/lite:framework",
        "//machina/lite:kernel_api",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/delegates/utils/dummy_delegate",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite(
    name = "simple_opaque_delegate",
    srcs = ["simple_opaque_delegate.cc"],
    hdrs = ["simple_opaque_delegate.h"],
    generate_opaque_delegate_target = True,
    tflite_deps = [
        "//machina/lite/c:c_api",
        "//machina/lite/c:c_api_experimental",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
    ],
    deps = [
        "//machina/lite:array",
        "//machina/lite:builtin_ops",
        "//machina/lite:minimal_logging",
        "//machina/lite:util",
        "//machina/lite/kernels/internal:compatibility",
    ],
)

filegroup(
    name = "c_api_test_builtin_op_models",
    testonly = 1,
    srcs = [
        "//machina/lite:testdata/add.bin",
        "//machina/lite:testdata/conv_huge_im2col.bin",
        "//machina/lite:testdata/multi_add.bin",
    ],
)

cc_test(
    name = "simple_opaque_delegate_test",
    srcs = ["simple_opaque_delegate_test.cc"],
    data = [":c_api_test_builtin_op_models"],
    deps = [
        ":simple_opaque_delegate",
        "//machina/lite:builtin_ops",
        "//machina/lite:framework_stable",
        "//machina/lite/c:c_api",
        "//machina/lite/c:c_api_experimental",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/delegates:delegate_test_util",
        "//machina/lite/delegates/utils/experimental/sample_stable_delegate",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels/internal:compatibility",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library_with_tflite(
    name = "async_type_helpers",
    srcs = ["async_type_helpers.cc"],
    hdrs = ["async_type_helpers.h"],
    tflite_deps = [
        ":ret_macros",
        "//machina/lite/async/interop/c:attribute_map",
        "//machina/lite/async/interop/c:constants",
        "//machina/lite/async/interop/c:types",
    ],
)

cc_library_with_tflite(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    tflite_deps = [
        "//machina/lite/c:c_api_experimental",
        "//machina/lite/c:c_api_types",
        ":ret_macros",
    ],
    deps = [
        "//machina/lite:array",
        "//machina/lite:minimal_logging",
        "@com_google_absl//absl/status",
    ],
)

cc_library_with_tflite(
    name = "ret_macros",
    srcs = [],
    hdrs = ["ret_macros.h"],
    tflite_deps = ["//machina/lite/c:c_api_types"],
    deps = [
        "//machina/lite:minimal_logging",
    ],
)

cc_library_with_tflite(
    name = "sync_fence",
    srcs = ["sync_fence.cc"],
    hdrs = ["sync_fence.h"],
    tflite_deps = [
        ":ret_macros",
    ],
    deps = [
        "//machina/lite:minimal_logging",
        "@com_google_absl//absl/types:span",
    ],
)
