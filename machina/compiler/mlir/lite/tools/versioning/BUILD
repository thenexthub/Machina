load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "versioning",
    srcs = [
        "op_version.cc",
        "runtime_version.cc",
    ],
    hdrs = [
        "op_version.h",
        "runtime_version.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":op_signature",
        "//machina/compiler/mlir/lite/core/c:tflite_common",
        "//machina/compiler/mlir/lite/kernels/internal:compatibility_macros",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "//machina/compiler/mlir/lite/schema:schema_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "op_version_test",
    srcs = [
        "op_version_test.cc",
    ],
    deps = [
        ":op_signature",
        ":versioning",
        "//machina/compiler/mlir/lite/core/c:tflite_common",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/lite/schema:schema_fbs_with_mutable",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "runtime_version_test",
    srcs = [
        "runtime_version_test.cc",
    ],
    deps = [
        ":versioning",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "op_signature",
    srcs = [
        "op_signature.cc",
    ],
    hdrs = [
        "op_signature.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/lite/core/api:flatbuffer_conversions",
        "//machina/compiler/mlir/lite/core/c:tflite_common",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/lite/schema:schema_utils",
        "@flatbuffers//:runtime_cc",
    ],
)

tf_cc_test(
    name = "op_signature_test",
    srcs = [
        "op_signature_test.cc",
    ],
    data = [
        "//machina/compiler/mlir/lite:testdata/add.bin",
        "//machina/compiler/mlir/lite:testdata/multi_signatures.bin",
    ],
    deps = [
        ":op_signature",
        "//machina/compiler/mlir/lite/core:absl_error_model_builder",
        "//machina/compiler/mlir/lite/core/c:tflite_common",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/core/platform:resource_loader",
        "@com_google_googletest//:gtest_main",
    ],
)
