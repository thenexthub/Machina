load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "interpreter_utils",
    testonly = 1,
    srcs = ["interpreter_utils.cc"],
    hdrs = ["interpreter_utils.h"],
    deps = [
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/api",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/schema:schema_fbs",
    ],
)
