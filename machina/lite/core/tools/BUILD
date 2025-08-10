load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/lite:build_def.bzl", "tflite_self_contained_libs_test_suite")
load("//machina/lite:special_rules.bzl", "verifier_internal_visibility_allowlist")
load("//machina/lite/core:special_rules.bzl", "verifier_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "verifier",
    srcs = ["verifier.cc"],
    hdrs = ["verifier.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//machina/lite:__subpackages__",
    ] + verifier_visibility_allowlist(),
    deps = [
        "//machina/compiler/mlir/lite/schema:schema_utils",
        "//machina/lite:framework",
        "//machina/lite:schema_fbs_version",
        "//machina/lite:util",
        "//machina/lite/c:common",
        "//machina/lite/core:framework",
        "//machina/lite/core/api:error_reporter",
        "//machina/lite/core/api:op_resolver",
        "//machina/lite/core/c:common",
        "//machina/lite/core/tools:verifier_internal",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:optional",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "verifier_test",
    size = "small",
    srcs = ["verifier_test.cc"],
    data = [
        "//machina/lite:testdata/sparse_tensor.bin",
    ],
    tags = [
        "tflite_not_portable",
    ],
    deps = [
        ":verifier",
        "//machina/compiler/mlir/lite/schema:schema_conversion_utils",
        "//machina/core/platform:resource_loader",
        "//machina/lite:framework",
        "//machina/lite:schema_fbs_version",
        "//machina/lite:string",
        "//machina/lite:util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/core:framework",
        "//machina/lite/core/api",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "verifier_internal",
    srcs = ["verifier_internal.cc"],
    hdrs = ["verifier_internal.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = verifier_internal_visibility_allowlist() + ["//machina/lite/tools:__subpackages__"],
    deps = [
        "//machina/lite/schema:schema_fbs",
        "@flatbuffers//:runtime_cc",
    ],
)

cc_test(
    name = "verifier_internal_test",
    size = "small",
    srcs = ["verifier_internal_test.cc"],
    deps = [
        ":verifier_internal",
        "//machina/compiler/mlir/lite/schema:schema_conversion_utils",
        "//machina/lite:framework",
        "//machina/lite:schema_fbs_version",
        "//machina/lite/c:common",
        "//machina/lite/core:framework",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

tflite_self_contained_libs_test_suite(name = "self_contained_libs_test_suite")
