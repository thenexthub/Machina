load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_test(
    name = "gpu_delegate_compatibility_checker_test",
    srcs = [
        "gpu_delegate_compatibility_checker_test.cc",
    ],
    data = [
        "//machina/lite:testdata/add.bin",
        "//machina/lite:testdata/conv3d_huge_im2col.bin",
    ],
    tags = ["no_oss"],  #TODO(b/276295784): Re-enable when fixed.
    deps = [
        ":gpu_delegate_compatibility_checker",
        "//machina/core/platform:resource_loader",
        "//machina/lite:model_builder",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_delegate_compatibility_checker",
    srcs = ["gpu_delegate_compatibility_checker.cc"],
    hdrs = [
        "gpu_delegate_compatibility_checker.h",
    ],
    deps = [
        "//machina/lite:model_builder",
        "//machina/lite/tools/delegates/compatibility/common:delegate_compatibility_checker_base",
        "//machina/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "//machina/lite/tools/versioning:gpu_compatibility",
        "//machina/lite/tools/versioning:op_signature",
        "@com_google_absl//absl/status",
    ],
)
