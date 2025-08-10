load(
    "//machina:machina.bzl",
    "if_libtpu",
    "tf_cuda_cc_test",
)
load("//machina:machina.default.bzl", "filegroup")
load(
    "//machina/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

# Library of gradient functions.
package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "array_grad",
    srcs = ["array_grad.cc"],
    hdrs = [
        "array_grad.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        "//machina/c/eager:abstract_context",
        "//machina/c/eager:gradients_internal",
    ],
)

cc_library(
    name = "math_grad",
    srcs = ["math_grad.cc"],
    hdrs = [
        "math_grad.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:gradients_internal",
        "//machina/c/experimental/ops:array_ops",
        "//machina/c/experimental/ops:math_ops",
        "//machina/c/experimental/ops:nn_ops",
    ],
)

cc_library(
    name = "nn_grad",
    srcs = ["nn_grad.cc"],
    hdrs = [
        "nn_grad.h",
    ],
    visibility = [
        "//machina/python/framework/experimental:__pkg__",
    ],
    deps = [
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:gradients_internal",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/c/experimental/ops:array_ops",
        "//machina/c/experimental/ops:math_ops",
        "//machina/c/experimental/ops:nn_ops",
        "//machina/core/lib/llvm_rtti",
        "//machina/core/platform:errors",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "not_differentiable",
    srcs = ["not_differentiable.cc"],
    hdrs = [
        "not_differentiable.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        "//machina/c/eager:abstract_context",
        "//machina/c/eager:gradients_internal",
    ],
)

cc_library(
    name = "gradients",
    hdrs = [
        "array_grad.h",
        "math_grad.h",
        "nn_grad.h",
        "not_differentiable.h",
    ],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":array_grad",
        ":math_grad",
        ":nn_grad",
        ":not_differentiable",
        "//machina/c/eager:abstract_context",
        "//machina/c/eager:gradients_internal",
    ],
)

tf_cuda_cc_test(
    name = "custom_gradient_test",
    size = "small",
    srcs = [
        "custom_gradient_test.cc",
    ],
    args = ["--heap_check="],  # TODO(b/174752220): Remove
    tags = tf_cuda_tests_tags(),
    deps = [
        "//machina/c:tf_status_helper",
        "//machina/c/eager:abstract_context",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/c/eager:c_api_unified_internal",
        "//machina/c/eager:gradients_internal",
        "//machina/c/eager:unified_api_testutil",
        "//machina/c/experimental/ops",
        "//machina/compiler/mlir/machina/c:mlir_c_api_registration",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:errors",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "array_grad.h",
        "math_grad.h",
        "nn_grad.h",
        "not_differentiable.h",
    ],
    visibility = ["//machina/python:__pkg__"],
)

cc_library(
    name = "grad_test_helper",
    testonly = True,
    srcs = ["grad_test_helper.cc"],
    hdrs = ["grad_test_helper.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/c/eager:gradient_checker",
        "//machina/c/eager:gradients_internal",
        "//machina/c/eager:unified_api_testutil",
        "//machina/c/experimental/gradients/tape:tape_context",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

tf_cuda_cc_test(
    name = "nn_grad_test",
    size = "small",
    srcs = [
        "nn_grad_test.cc",
    ],
    args = ["--heap_check="],  # TODO(b/174752220): Remove
    tags = tf_cuda_tests_tags() + ["no_cuda_asan"],  # b/173654156,
    deps = [
        ":grad_test_helper",
        ":nn_grad",
        "//machina/c:tf_status_helper",
        "//machina/c/eager:c_api_test_util",
        "//machina/c/eager:c_api_unified_internal",
        "//machina/c/eager:unified_api_testutil",
        "//machina/c/experimental/gradients/tape:tape_context",
        "//machina/c/experimental/ops:nn_ops",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:tensor_float_32_utils",
    ] + if_libtpu(
        if_false = ["//machina/compiler/mlir/machina/c:mlir_c_api_registration"],
        if_true = [],
    ),
)

tf_cuda_cc_test(
    name = "math_grad_test",
    size = "small",
    srcs = [
        "math_grad_test.cc",
    ],
    args = ["--heap_check="],  # TODO(b/174752220): Remove
    tags = tf_cuda_tests_tags() + ["no_cuda_asan"],  # b/173654156,
    deps = [
        ":grad_test_helper",
        ":math_grad",
        "//machina/c:tf_status_helper",
        "//machina/c/eager:c_api_test_util",
        "//machina/c/eager:c_api_unified_internal",
        "//machina/c/eager:unified_api_testutil",
        "//machina/c/experimental/gradients/tape:tape_context",
        "//machina/c/experimental/ops:math_ops",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:tensor_float_32_utils",
    ] + if_libtpu(
        if_false = ["//machina/compiler/mlir/machina/c:mlir_c_api_registration"],
        if_true = [],
    ),
)

tf_cuda_cc_test(
    name = "array_grad_test",
    size = "small",
    srcs = [
        "array_grad_test.cc",
    ],
    args = ["--heap_check="],  # TODO(b/174752220): Remove
    tags = tf_cuda_tests_tags() + ["no_cuda_asan"],  # b/173654156,
    deps = [
        ":array_grad",
        ":grad_test_helper",
        "//machina/c:tf_status_helper",
        "//machina/c/eager:c_api_test_util",
        "//machina/c/eager:c_api_unified_internal",
        "//machina/c/eager:unified_api_testutil",
        "//machina/c/experimental/gradients/tape:tape_context",
        "//machina/c/experimental/ops:array_ops",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:tensor_float_32_utils",
    ] + if_libtpu(
        if_false = ["//machina/compiler/mlir/machina/c:mlir_c_api_registration"],
        if_true = [],
    ),
)
