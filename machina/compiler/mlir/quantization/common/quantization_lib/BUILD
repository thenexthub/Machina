load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    # By default, these targets should only be used within the quantization library.
    default_visibility = [
        "//learning/brain/mlir/quantization:__subpackages__",
        "//platforms/darwinn/compiler:__subpackages__",
        "//machina:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "portable_tensor_utils",
    srcs = ["portable_tensor_utils.cc"],
    hdrs = ["portable_tensor_utils.h"],
)

cc_library(
    name = "quantization_lib",
    srcs = [
        "quantization_driver.cc",
        "quantization_interface.cc.inc",
        "quantization_utils.cc",
    ],
    hdrs = [
        "quantization_driver.h",
        "quantization_interface.h.inc",
        "quantization_traits.h",
        "quantization_utils.h",
    ],
    deps = [
        ":portable_tensor_utils",
        ":quantization_config",
        ":quantization_interfaces_inc_gen",
        "//machina/compiler/mlir/quantization/common/ir:QuantOps",
        "//machina/compiler/mlir/tools/optimize:quantization_utils",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "quantization_driver_test",
    srcs = ["quantization_driver_test.cc"],
    deps = [
        ":quantization_lib",
        "//machina/compiler/mlir/quantization/common:attrs_and_constraints",
        "//machina/compiler/mlir/quantization/common:func",
        "//machina/compiler/mlir/quantization/common:test_base",
        "//machina/compiler/mlir/quantization/common/ir:QuantOps",
        "//machina/compiler/mlir/machina",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "quantization_config",
    srcs = [
        "quantization_config.cc",
    ],
    hdrs = [
        "quantization_config.h",
    ],
    deps = [
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
    ],
)

td_library(
    name = "quantization_td_files",
    srcs = [
        "quantization.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/common/ir:QuantizationOpsTdFiles",
        "@toolchain-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "quantization_interfaces_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "quantization_interface.h.inc": ["-gen-op-interface-decls"],
        "quantization_interface.cc.inc": ["-gen-op-interface-defs"],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "quantization.td",
    deps = [
        ":quantization_td_files",
    ],
)

exports_files([
    "quantization_traits.h",
    "quantization_config.h",
    "quantization_utils.h",
])
