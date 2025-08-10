load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/compiler/mlir/quantization/machina:internal_visibility_allowlist_package",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "tf_op_quant_spec",
    srcs = [
        "tf_op_quant_spec.cc",
    ],
    hdrs = ["tf_op_quant_spec.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/common/quantization_lib",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/machina",
        "@com_google_absl//absl/container:flat_hash_set",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "tf_op_quant_spec_test",
    srcs = ["tf_op_quant_spec_test.cc"],
    deps = [
        ":tf_op_quant_spec",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tf_quantize_op",
    srcs = [
        "tf_quantize_op.cc",
    ],
    hdrs = ["tf_quantize_op.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/common/quantization_lib",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/quantization/machina/utils:tf_quantize_op_utils",
        "//machina/compiler/mlir/machina",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "tf_quantize_op_test",
    srcs = ["tf_quantize_op_test.cc"],
    deps = [
        ":tf_quantize_op",
        "//machina/compiler/mlir/quantization/common:attrs_and_constraints",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_types",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "uniform_op_quant_spec",
    srcs = [
        "uniform_op_quant_spec.cc",
    ],
    hdrs = ["uniform_op_quant_spec.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/common/quantization_lib",
        "//machina/compiler/mlir/machina",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)
