load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:private",
    ],
)

gentbl_cc_library(
    name = "inference_passes_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"inference_passes.h.inc": [
        "-gen-pass-decls",
        "-name=TF2XLA",
    ]},
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "inference_passes.td",
    deps = [
        "@toolchain-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "inference_metrics_pass",
    srcs = ["inference_metrics_pass.cc"],
    hdrs = ["inference_passes.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":inference_passes_inc_gen",
        "//machina/compiler/mlir/machina",
        "//machina/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FuncExtensions",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "inference_metrics_pass_test",
    srcs = ["inference_metrics_pass_test.cc"],
    deps = [
        "inference_metrics_pass",
        "//machina/compiler/mlir/tf2xla/transforms:test_utils",
        "//machina/core/lib/monitoring:cell_reader",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)
