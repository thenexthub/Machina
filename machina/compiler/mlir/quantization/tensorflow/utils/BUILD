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
    name = "fake_quant_utils",
    srcs = ["fake_quant_utils.cc"],
    hdrs = [
        "fake_quant_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/common/ir:QuantOps",
        "//machina/compiler/mlir/quantization/common/quantization_lib",
        "//machina/compiler/mlir/machina:machina_ops",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Dialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tf_quantize_op_utils",
    srcs = ["tf_quantize_op_utils.cc"],
    hdrs = ["tf_quantize_op_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tf_to_uniform_attribute_utils",
    srcs = ["tf_to_uniform_attribute_utils.cc"],
    hdrs = ["tf_to_uniform_attribute_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/common:attrs_and_constraints",
        "//machina/compiler/mlir/quantization/common/quantization_lib",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/quantization/machina/ops:uniform_op_quant_spec",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "tf_to_uniform_attribute_utils_test",
    srcs = ["tf_to_uniform_attribute_utils_test.cc"],
    deps = [
        ":tf_to_uniform_attribute_utils",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_types",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "tf_to_xla_attribute_utils",
    srcs = ["tf_to_xla_attribute_utils.cc"],
    hdrs = ["tf_to_xla_attribute_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/kernels:padding",
        "//machina/compiler/mlir/quantization/common:attrs_and_constraints",
        "//machina/compiler/mlir/quantization/machina/cc:constant_fold",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings:str_format",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla:xla_data_proto_cc",
    ],
)

tf_cc_test(
    name = "tf_to_xla_attribute_utils_test",
    srcs = ["tf_to_xla_attribute_utils_test.cc"],
    deps = [
        ":tf_to_xla_attribute_utils",
        "//machina/c/eager:c_api",
        "//machina/compiler/mlir/quantization/common:attrs_and_constraints",
        "//machina/compiler/mlir/machina:machina_types",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)
