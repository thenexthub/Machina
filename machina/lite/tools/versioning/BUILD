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
    hdrs = [
        "op_version.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/lite/tools/versioning",
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
        "//machina/compiler/mlir/lite/tools/versioning:op_signature",
        "//machina/lite:stderr_reporter",
        "//machina/lite/core/api",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/schema:schema_fbs",
        "//machina/lite/schema:schema_utils",
    ],
)

tf_cc_test(
    name = "op_signature_test",
    srcs = [
        "op_signature_test.cc",
    ],
    data = [
        "//machina/lite:testdata/add.bin",
        "//machina/lite:testdata/multi_signatures.bin",
    ],
    deps = [
        ":op_signature",
        "//machina/core/platform:resource_loader",
        "//machina/lite:builtin_ops",
        "//machina/lite/core:model_builder",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "gpu_compatibility",
    srcs = [
        "gpu_compatibility.cc",
    ],
    hdrs = [
        "gpu_compatibility.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":op_signature",
        "//machina/lite:builtin_op_data",
        "//machina/lite:builtin_ops",
        "//machina/lite:util",
        "//machina/lite/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/schema:schema_fbs",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "gpu_compatibility_test",
    srcs = [
        "gpu_compatibility_test.cc",
    ],
    data = [
        "//machina/lite:testdata/conv3d_huge_im2col.bin",
        "//machina/lite:testdata/conv_huge_im2col.bin",
        "//machina/lite:testdata/multi_add_flex.bin",
    ],
    deps = [
        ":gpu_compatibility",
        ":op_signature",
        "//machina/core/platform:resource_loader",
        "//machina/lite/core:model_builder",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels/internal:types",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
