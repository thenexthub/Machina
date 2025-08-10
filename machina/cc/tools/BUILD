#include "third_party/absl/strings/str_cat.h"
#Description:
# TensorFlow cc tools.

load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "freeze_saved_model",
    srcs = ["freeze_saved_model.cc"],
    hdrs = ["freeze_saved_model.h"],
    deps = [
        "//machina/cc/saved_model:loader",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "freeze_saved_model_test",
    srcs = ["freeze_saved_model_test.cc"],
    deps = [
        ":freeze_saved_model",
        "//machina/cc:cc_ops",
        "//machina/cc:ops",
        "//machina/cc:resource_variable_ops",
        "//machina/cc:scope",
        "//machina/cc/saved_model:loader",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:errors",
    ],
)
