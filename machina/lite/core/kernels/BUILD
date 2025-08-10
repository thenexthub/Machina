load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/lite:build_def.bzl", "tflite_copts", "tflite_self_contained_libs_test_suite")
load("//machina/lite/core:special_rules.bzl", "builtin_ops_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

exports_files(
    srcs = [
        "builtin_op_kernels.h",
        "register.h",
    ],
    visibility = [
        "//research/drishti/benchmarking/async:__subpackages__",
        "//machina/lite:__subpackages__",
    ],
)

cc_test(
    name = "builtin_ops_test",
    srcs = ["register_test.cc"],
    deps = [
        ":builtin_ops",
        "//machina/compiler/mlir/lite/tools/versioning",
        "//machina/lite:mutable_op_resolver",
        "//machina/lite/c:common",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "builtin_ops",
    srcs = ["register.cc"],
    hdrs = [
        "builtin_op_kernels.h",
        "register.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//machina/lite:__subpackages__",
    ] + builtin_ops_visibility_allowlist(),
    deps = [
        "//machina/lite:mutable_op_resolver",
        "//machina/lite:tflite_with_xnnpack_optional",
        "//machina/lite/core:cc_api_stable",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:builtin_op_kernels",
        "//machina/lite/schema:schema_fbs",
    ],
)

# For internal usage by shared libraries only.
cc_library(
    name = "builtin_ops_all_linked",
    srcs = ["register.cc"],
    hdrs = [
        "builtin_op_kernels.h",
        "register.h",
    ],
    copts = tflite_copts(),
    # Limit visibility to TFLite only.
    visibility = [
        "//machina/lite:__subpackages__",
    ],
    deps = [
        "//machina/lite:framework_stable",
        "//machina/lite:tflite_with_xnnpack_optional",
        "//machina/lite/core:cc_api_stable",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:builtin_op_kernels",
        "//machina/lite/schema:schema_fbs",
    ],
    alwayslink = 1,
)

tflite_self_contained_libs_test_suite(name = "self_contained_libs_test_suite")
