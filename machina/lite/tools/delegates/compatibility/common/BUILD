load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "online_helper_delegate",
    srcs = ["online_helper_delegate.cc"],
    hdrs = ["online_helper_delegate.h"],
    deps = [
        "//machina/lite:kernel_api",
        "//machina/lite:minimal_logging",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core/c:common",
        "//machina/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "delegate_compatibility_checker_base",
    srcs = ["delegate_compatibility_checker_base.cc"],
    hdrs = [
        "delegate_compatibility_checker_base.h",
    ],
    deps = [
        ":delegate_compatibility_checker_util",
        "//machina/lite:framework_stable",
        "//machina/lite/core:framework_stable",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:builtin_ops",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "//machina/lite/tools/versioning:op_signature",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "delegate_compatibility_checker_util",
    hdrs = [
        "delegate_compatibility_checker_util.h",
    ],
)
