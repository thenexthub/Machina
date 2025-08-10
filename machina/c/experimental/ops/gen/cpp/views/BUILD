load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_library(
    name = "views",
    srcs = glob(
        ["*.cc"],
        exclude = ["*_test.cc"],
    ),
    hdrs = glob(["*.h"]),
    visibility = [
        "//machina/c/experimental/ops/gen/cpp:__pkg__",
        "//machina/c/experimental/ops/gen/cpp/renderers:__pkg__",
    ],
    deps = [
        "//machina/c/experimental/ops/gen/common",
        "//machina/c/experimental/ops/gen/model",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)
