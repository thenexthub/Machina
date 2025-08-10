load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

cc_library(
    name = "outline_operations",
    srcs = ["outline_operations.cc"],
    hdrs = ["outline_operations.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/machina:cluster_util",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ],
)
