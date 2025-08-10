load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/compiler/mlir/lite:build_def.bzl", "tflite_copts_warnings")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

exports_files(
    [
        "model_builder_base.h",
    ],
    visibility = ["//machina/lite/core:__pkg__"],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "model_builder_base",
    srcs = ["model_builder_base.cc"],
    hdrs = ["model_builder_base.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    visibility = ["//visibility:public"],
    deps = [
        ":macros",
        "//machina/compiler/mlir/lite:allocation",
        "//machina/compiler/mlir/lite/core/api:error_reporter",
        "//machina/compiler/mlir/lite/core/api:verifier",
        "//machina/compiler/mlir/lite/schema:schema_fbs",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "absl_error_model_builder",
    srcs = ["absl_error_model_builder.cc"],
    hdrs = ["absl_error_model_builder.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts_warnings(),
    visibility = ["//visibility:public"],
    deps = [
        ":model_builder_base",
        "//machina/compiler/mlir/lite/core/api:error_reporter",
        "@com_google_absl//absl/log",
    ],
)
