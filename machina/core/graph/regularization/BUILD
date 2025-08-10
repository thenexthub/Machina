# Description:
# A TensorFlow Graph Regularization Library

load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load("//machina:machina.default.bzl", "filegroup")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "test_files",
    srcs = glob([
        "testdata/bert2/**",
        "testdata/bert1/**",
    ]),
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:numbers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "util_test",
    size = "small",
    srcs = [
        "util_test.cc",
    ],
    deps = [
        ":util",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "simple_delete",
    srcs = ["simple_delete.cc"],
    hdrs = ["simple_delete.h"],
    deps = [
        ":util",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:op_types",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "simple_delete_test",
    size = "small",
    srcs = [
        "simple_delete_test.cc",
    ],
    data = [
        ":test_files",
    ],
    deps = [
        ":simple_delete",
        ":util",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core/platform:path",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)
