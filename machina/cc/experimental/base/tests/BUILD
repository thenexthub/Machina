# Tests for the C++ header-only base types.
load("//machina:machina.bzl", "tf_cc_test")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "tensor_types_test_util",
    testonly = True,
    hdrs = ["tensor_types_test_util.h"],
    deps = [
        "//machina/c:tf_datatype",
    ],
)

tf_cc_test(
    name = "tensor_test",
    srcs = [
        "tensor_test.cc",
    ],
    deps = [
        ":tensor_types_test_util",
        "//machina/c:tf_datatype",
        "//machina/cc/experimental/base/public:status",
        "//machina/cc/experimental/base/public:tensor",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "tensorhandle_test",
    srcs = [
        "tensorhandle_test.cc",
    ],
    deps = [
        ":tensor_types_test_util",
        "//machina/c:tf_datatype",
        "//machina/cc/experimental/base/public:runtime",
        "//machina/cc/experimental/base/public:runtime_builder",
        "//machina/cc/experimental/base/public:status",
        "//machina/cc/experimental/base/public:tensor",
        "//machina/cc/experimental/base/public:tensorhandle",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
