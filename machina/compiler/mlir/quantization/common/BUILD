load("@toolchain-project//mlir:tblgen.bzl", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    # By default, these targets should only be used within the quantization library.
    default_visibility = [
        "//learning/brain/mlir/quantization:__subpackages__",
        "//machina/compiler/mlir/lite:__subpackages__",
        "//machina/compiler/mlir/quantization:__subpackages__",
        "//machina/compiler/mlir/stablehlo:__subpackages__",
    ],
    licenses = ["notice"],
)

td_library(
    name = "lift_as_function_call_td_files",
    srcs = [
        "lift_as_function_call.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@toolchain-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "lift_as_function_call",
    srcs = ["lift_as_function_call.cc"],
    hdrs = ["lift_as_function_call.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":attrs_and_constraints",
        "//machina/compiler/mlir/quantization/common/quantization_lib",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo:stablehlo_type_utils",
        "//machina/compiler/mlir/quantization/machina/cc:quantization_unit_loc",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_ops",
        "//machina/compiler/mlir/machina:xla_call_module_attrs",
        "//machina/core:framework_lite",
        "//machina/core/ir/types:Dialect",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@stablehlo//:version",
    ],
)

tf_cc_test(
    name = "lift_as_function_call_test",
    srcs = ["lift_as_function_call_test.cc"],
    deps = [
        ":attrs_and_constraints",
        ":func",
        ":lift_as_function_call",
        ":test_base",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/machina",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "func",
    srcs = ["func.cc"],
    hdrs = ["func.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//machina/cc/saved_model:signature_constants",
        "//machina/compiler/mlir/machina:import_model",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "func_test",
    srcs = ["func_test.cc"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":func",
        ":test_base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
    ],
)

cc_library(
    name = "test_base",
    testonly = 1,
    srcs = [],
    hdrs = ["test_base.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":func",
        "//machina/compiler/mlir/quantization/common/ir:QuantOps",
        "//machina/compiler/mlir/quantization/stablehlo/cc:context",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:machina_types",
        "//machina/core:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "attrs_and_constraints",
    srcs = [
        "attrs_and_constraints.cc",
    ],
    hdrs = [
        "attrs_and_constraints.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":tf_uniform_quantized_types",
        "//machina/compiler/mlir/quantization/common/quantization_lib",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina:xla_call_module_attrs",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

tf_cc_test(
    name = "attrs_and_constraints_test",
    srcs = ["attrs_and_constraints_test.cc"],
    deps = [
        ":attrs_and_constraints",
        ":func",
        ":test_base",
        "//machina/compiler/mlir/machina",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@stablehlo//:stablehlo_ops",
    ],
)

td_library(
    name = "quant_td_files",
    srcs = [
        "attrs_and_constraints.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":lift_as_function_call_td_files",
        "//machina/compiler/mlir/machina:machina_ops_td_files",
        "@toolchain-project//mlir:ArithOpsTdFiles",
        "@toolchain-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "tf_uniform_quantized_types",
    srcs = ["tf_uniform_quantized_types.cc"],
    hdrs = ["tf_uniform_quantized_types.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "uniform_quantized_types",
    srcs = ["uniform_quantized_types.cc"],
    hdrs = ["uniform_quantized_types.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "uniform_quantized_types_test",
    srcs = ["uniform_quantized_types_test.cc"],
    deps = [
        ":test_base",
        ":uniform_quantized_types",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)
