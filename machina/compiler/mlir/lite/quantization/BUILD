load("//machina:machina.bzl", "tf_cc_test", "tf_native_cc_binary")
load("//machina/core/platform:build_config.bzl", "tf_proto_library")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    # copybara:uncomment_begin(google-only)
    # default_visibility = [
    # ":friends",
    # "//machina:__pkg__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    default_visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//learning/brain/mlir/quantization/...",
        "//machina/compiler/mlir/...",
        "//machina/lite/tools/optimize/...",
    ],
)

tf_proto_library(
    name = "quantization_info_proto",
    srcs = [
        "quantization_info.proto",
    ],
)

cc_library(
    name = "quantization_passes",
    srcs = [
        "import_quant_stats_pass.cc",
    ],
    hdrs = [
        "quantization_passes.h",
    ],
    deps = [
        ":quantization_info_proto_cc",
        "//machina/compiler/mlir/lite/quantization/ir:QuantOps",
        "//machina/compiler/mlir/machina:import_utils",
        "//machina/compiler/mlir/machina:location_utils",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
    ],
    alwayslink = 1,
)

tf_native_cc_binary(
    name = "op_quant_spec_getters_gen",
    srcs = [
        "tools/op_quant_spec_getters_gen.cc",
    ],
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:TableGen",
        "@toolchain-project//mlir:TableGen",
    ],
)

tf_native_cc_binary(
    name = "tflite_op_coverage_spec_getters_gen",
    srcs = [
        "tools/tflite_op_coverage_spec_getters_gen.cc",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//toolchain:TableGen",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:TableGen",
        "@local_tsl//tsl/platform:regexp",
        "@local_xla//xla/tsl/platform:logging",
    ],
)

cc_library(
    name = "numerical_utils",
    srcs = ["numerical_utils.cc"],
    hdrs = ["numerical_utils.h"],
    deps = [
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "device_target",
    srcs = ["device_target.cc"],
    hdrs = ["device_target.h"],
    deps = [
        ":numerical_utils",
        "//machina/compiler/mlir/lite/quantization/ir:QuantOps",
        "@com_google_absl//absl/types:optional",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "quantization_context",
    srcs = ["quantization_context.cc"],
    hdrs = ["quantization_context.h"],
    deps = [
        ":device_target",
        "//machina/compiler/mlir/lite/quantization/common/quantization_lib",
        "//machina/compiler/mlir/lite/quantization/ir:QuantOps",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:QuantOps",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "numerical_utils_test",
    srcs = ["numerical_utils_test.cc"],
    deps = [
        ":numerical_utils",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
    ],
)
