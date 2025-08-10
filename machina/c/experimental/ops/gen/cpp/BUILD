load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_library(
    name = "cpp",
    srcs = glob(
        ["*.cc"],
        exclude = ["*_test.cc"],
    ),
    hdrs = glob(["*.h"]),
    visibility = ["//machina/c/experimental/ops/gen:__pkg__"],
    deps = [
        "//machina/c/experimental/ops/gen/common",
        "//machina/c/experimental/ops/gen/cpp/renderers",
        "//machina/c/experimental/ops/gen/cpp/views",
        "//machina/c/experimental/ops/gen/model",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:op_gen_lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:str_util",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "cpp_generator_test",
    size = "small",
    srcs = ["cpp_generator_test.cc"],
    data = ["//machina/c/experimental/ops/gen/cpp/golden"],
    deps = [
        ":cpp",
        "//machina/c/experimental/ops/gen/common",
        "//machina/c/experimental/ops/gen/cpp/renderers",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@local_xla//xla/tsl/platform:status",
    ],
)
