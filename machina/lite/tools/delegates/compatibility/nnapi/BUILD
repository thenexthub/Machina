load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# BUILD rules for NNAPI delegate compatibility checking.
package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "nnapi_compatibility_lib",
    srcs = [
        "nnapi_compatibility_lib.cc",
    ],
    hdrs = [
        "nnapi_compatibility_lib.h",
    ],
    deps = [
        "//machina/lite:framework_stable",
        "//machina/lite:minimal_logging",
        "//machina/lite/core/c:common",
        "//machina/lite/delegates/nnapi:nnapi_delegate",
    ],
)

cc_library(
    name = "nnapi_delegate_compatibility_checker",
    srcs = ["nnapi_delegate_compatibility_checker.cc"],
    hdrs = [
        "nnapi_delegate_compatibility_checker.h",
    ],
    copts = ["-DNNAPI_VERBOSE_VALIDATION"],
    deps = [
        "//machina/lite:framework_stable",
        "//machina/lite/c:common",
        "//machina/lite/core:cc_api_stable",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/delegates/nnapi:nnapi_delegate",
        "//machina/lite/nnapi:nnapi_lib",
        "//machina/lite/tools/delegates/compatibility/common:delegate_compatibility_checker_base",
        "//machina/lite/tools/delegates/compatibility/common:delegate_compatibility_checker_util",
        "//machina/lite/tools/delegates/compatibility/common:online_helper_delegate",
        "//machina/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "//machina/lite/tools/versioning:op_signature",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "nnapi_compatibility_lib_test",
    srcs = [
        "nnapi_compatibility_lib_test.cc",
    ],
    deps = [
        ":nnapi_compatibility_lib",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/delegates/nnapi:nnapi_delegate_verbose_validation",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "nnapi_delegate_compatibility_checker_test",
    srcs = [
        "nnapi_delegate_compatibility_checker_test.cc",
    ],
    data = [
        "//machina/lite:testdata/add.bin",
    ],
    tags = ["no_oss"],  #TODO(b/276295784): Re-enable when fixed.
    deps = [
        ":nnapi_delegate_compatibility_checker",
        "//machina/core/platform:resource_loader",
        "//machina/lite:model_builder",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/tools/delegates/compatibility/protos:compatibility_result_cc",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
