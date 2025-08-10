# Description:
#   Tensorflow / MLIR utils.

load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "name_utils",
    srcs = ["name_utils.cc"],
    hdrs = ["name_utils.h"],
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "string_container_utils",
    hdrs = ["string_container_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
    ],
)

cc_library(
    name = "array_container_utils",
    hdrs = ["array_container_utils.h"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
    ],
)

cc_library(
    name = "saved_model_converter_utils",
    srcs = ["saved_model_converter_utils.cc"],
    hdrs = ["saved_model_converter_utils.h"],
    visibility = [
        "//machina/cc/experimental/tfa:__subpackages__",
    ],
    deps = [
        "//machina/cc/saved_model:loader",
        "//machina/compiler/mlir/machina:mlir_import_options",
        "//machina/compiler/mlir/machina:translate_lib",
        "//machina/compiler/mlir/tf2xla/api/v2:mlir_roundtrip_flags",
        "//machina/core/framework:op",
        "//machina/core/framework:op_def_builder",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "validators",
    srcs = [
        "validators.cc",
    ],
    hdrs = [
        "validators.h",
    ],
    deps = [
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)
