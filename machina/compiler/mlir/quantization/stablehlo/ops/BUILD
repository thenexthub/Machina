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
    name = "stablehlo_op_quant_spec",
    srcs = [
        "stablehlo_op_quant_spec.cc",
    ],
    hdrs = ["stablehlo_op_quant_spec.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/common:attrs_and_constraints",
        "//machina/compiler/mlir/quantization/common:lift_as_function_call",
        "//machina/compiler/mlir/quantization/common/ir:QuantOps",
        "//machina/compiler/mlir/quantization/common/quantization_lib",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/machina",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:protobuf",
        "@stablehlo//:stablehlo_ops",
    ],
)

tf_cc_test(
    name = "stablehlo_op_quant_spec_test",
    srcs = ["stablehlo_op_quant_spec_test.cc"],
    deps = [
        ":stablehlo_op_quant_spec",
        "//machina/compiler/mlir/quantization/common:attrs_and_constraints",
        "//machina/compiler/mlir/quantization/common:func",
        "//machina/compiler/mlir/quantization/common:test_base",
        "//machina/compiler/mlir/quantization/common/ir:QuantOps",
        "//machina/compiler/mlir/quantization/common/quantization_lib",
        "//machina/compiler/mlir/machina",
        "//machina/core:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)
