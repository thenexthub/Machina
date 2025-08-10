load("//machina:machina.bzl", "tf_cc_test")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        ":friends",
        "//machina:__pkg__",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//learning/brain/experimental/mlir/...",
        "//machina/compiler/mlir/lite/...",
        "//machina/lite/...",
    ],
)

cc_library(
    name = "sparsify_model",
    srcs = [
        "sparsify_model.cc",
    ],
    hdrs = [
        "sparsify_model.h",
    ],
    deps = [
        "//machina/compiler/mlir/lite:flatbuffer_translate_lib",
        "//machina/compiler/mlir/lite:pass_registry_utils",
        "//machina/compiler/mlir/lite:machina_lite_d2s",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/lite/tools/optimize:reduced_precision_metadata",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "sparsify_model_test",
    srcs = ["sparsify_model_test.cc"],
    data = [
        ":testdata/sparse_tensor.bin",
    ],
    deps = [
        ":sparsify_model",
        "//machina/compiler/mlir/lite/core:absl_error_model_builder",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "//machina/compiler/mlir/lite/tools/optimize:reduced_precision_metadata",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)
