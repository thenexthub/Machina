load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "cache_buffer",
    srcs = ["cache_buffer.cc"],
    hdrs = [
        "cache_buffer.h",
        "//machina/lite/core/c:common.h",
    ],
    deps = [
        ":resource",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "cache_buffer_test",
    srcs = ["cache_buffer_test.cc"],
    deps = [
        ":cache_buffer",
        "//machina/lite/c:common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "resource",
    srcs = [
        "initialization_status.cc",
        "resource_variable.cc",
        "static_hashtable.cc",
    ],
    hdrs = [
        "initialization_status.h",
        "lookup_interfaces.h",
        "lookup_util.h",
        "resource_base.h",
        "resource_variable.h",
        "static_hashtable.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/lite:string_util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels/internal:tensor",
        "//machina/lite/kernels/internal:types",
    ],
)

cc_test(
    name = "resource_variable_test",
    srcs = [
        "resource_variable_test.cc",
    ],
    deps = [
        ":resource",
        "//machina/lite:util",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
