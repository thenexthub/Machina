# Description:
#   Provides ApiDef access and ApiDef validation for TensorFlow.
#
# The following targets can be used to access ApiDefs:
#   :base_api_def
#   :python_api_def
#   :java_api_def

load(
    "@local_config_tensorrt//:build_defs.bzl",
    "if_tensorrt",
)
load(
    "@local_xla//xla/tsl/mkl:build_defs.bzl",
    "if_mkl",
)
load(
    "//machina:machina.bzl",
    "tf_cc_binary",
    "tf_cc_test",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

alias(
    name = "base_api_def",
    actual = "//machina/core/api_def/base_api:base_api_def",
    visibility = ["//machina:internal"],
)

alias(
    name = "python_api_def",
    actual = "//machina/core/api_def/python_api:python_api_def",
    visibility = ["//machina:internal"],
)

alias(
    name = "java_api_def",
    actual = "//machina/core/api_def/java_api:java_api_def",
    visibility = ["//machina:internal"],
)

cc_library(
    name = "excluded_ops_lib",
    srcs = ["excluded_ops.cc"],
    hdrs = ["excluded_ops.h"],
    copts = if_mkl(["-DINTEL_MKL=1"]) + if_tensorrt(["-DGOOGLE_TENSORRT=1"]),
)

cc_library(
    name = "update_api_def_lib",
    srcs = ["update_api_def.cc"],
    hdrs = ["update_api_def.h"],
    deps = [
        ":excluded_ops_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:op_gen_lib",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "update_api_def_test",
    srcs = ["update_api_def_test.cc"],
    deps = [
        ":update_api_def_lib",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

tf_cc_binary(
    name = "update_api_def",
    srcs = [
        "update_api_def_main.cc",
    ],
    data = [
        ":base_api_def",
    ],
    deps = [
        ":update_api_def_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
    ],
)

tf_cc_test(
    name = "api_test",
    srcs = ["api_test.cc"],
    data = [
        ":base_api_def",
        ":python_api_def",
    ],
    deps = [
        ":excluded_ops_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:lib_test_internal",
        "//machina/core:op_gen_lib",
        "//machina/core:ops",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:resource_loader",
        "//machina/core/tpu/ops:sparse_core_ops",
        "//machina/core/tpu/ops:sparse_core_preprocess_ops",
        "//machina/core/tpu/ops:tpu_copy_with_dynamic_shape_op",
    ],
)
