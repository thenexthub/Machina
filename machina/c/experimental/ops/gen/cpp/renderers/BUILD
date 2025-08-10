load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "tf_cc_tests",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_library(
    name = "renderers",
    srcs = glob(
        ["*.cc"],
        exclude = ["*_test.cc"],
    ),
    hdrs = glob(["*.h"]),
    visibility = [
        "//machina/c/experimental/ops/gen:__pkg__",
        "//machina/c/experimental/ops/gen/cpp:__pkg__",
    ],
    deps = [
        "//machina/c/experimental/ops/gen/common",
        "//machina/c/experimental/ops/gen/cpp/views",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

tf_cc_tests(
    name = "renderer_tests",
    size = "small",
    srcs = glob(
        ["*_test.cc"],
    ),
    deps = [
        ":renderers",
        "//machina/c/experimental/ops/gen/common",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/platform:types",
    ],
)
