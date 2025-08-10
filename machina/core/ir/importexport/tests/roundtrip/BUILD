load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_cc_binary")

# Regression tests for bridge.
load(":roundtrip.bzl", "glob_roundtrip_tests")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/core/ir/importexport:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "roundtrip",
    srcs = [
        "roundtrip.cc",
    ],
    hdrs = ["roundtrip.h"],
    deps = [
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "@toolchain-project//toolchain:Support",
    ],
)

# This will run the `verify-roundtrip` binary below on every single pbtxt in this directory.
glob_roundtrip_tests(
    exclude = [],
    test_file_exts = ["pbtxt"],
)

tf_cc_binary(
    name = "verify-roundtrip",
    testonly = 1,
    srcs = ["verify_roundtrip.cc"],
    deps = [
        ":roundtrip",
        "//machina/compiler/mlir:init_mlir",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:ops",  # Ops need to be registered for import.
        "//machina/core/ir/importexport:graphdef_export",
        "//machina/core/ir/importexport:graphdef_import",
        "//machina/core/ir/importexport:load_proto",
        "//machina/core/transforms/consolidate_attrs:Pass",
        "@com_google_googletest//:gtest",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TranslateLib",
    ],
)
