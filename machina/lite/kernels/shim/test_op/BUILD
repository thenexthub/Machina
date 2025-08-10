# A simple op. for testing and demonstrating the OpKernel interface.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:machina.bzl", "tf_cc_test", "tf_kernel_library")
load("//machina/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/lite/kernels/shim:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "simple_op",
    hdrs = ["simple_op.h"],
    deps = [
        "//machina/lite/kernels/shim:op_kernel",
        "//machina/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_kernel_library(
    name = "simple_tf_op",
    srcs = ["simple_tf_op.cc"],
    hdrs = ["simple_tf_op.h"],
    deps = [
        ":simple_op",
        "//machina/core:framework",
        "//machina/lite/kernels/shim:tf_op_shim",
    ],
)

tf_cc_test(
    name = "simple_tf_op_test",
    srcs = ["simple_tf_op_test.cc"],
    tags = [
        "nochromiumos_arm",
        # Exclude from mobile builds
        "tflite_not_portable",
        "tflite_not_portable_android",
    ],
    deps = [
        ":simple_tf_op",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:testlib",
        "//machina/core/kernels:ops_testutil",
        "//machina/core/platform:tstring",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "simple_tflite_op",
    srcs = ["simple_tflite_op.cc"],
    hdrs = ["simple_tflite_op.h"],
    deps = [
        ":simple_op",
        "//machina/lite:mutable_op_resolver",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels/shim:tflite_op_shim",
    ],
)

cc_test(
    name = "simple_tflite_op_test",
    srcs = ["simple_tflite_op_test.cc"],
    deps = [
        ":simple_tflite_op",
        "//machina/core/platform:tstring",
        "//machina/lite/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "tmpl_op",
    hdrs = ["tmpl_op.h"],
    deps = [
        "//machina/lite/kernels/shim:op_kernel",
        "//machina/lite/kernels/shim:status_macros",
        "@com_google_absl//absl/status",
    ],
)

tf_kernel_library(
    name = "tmpl_tf_op",
    srcs = ["tmpl_tf_op.cc"],
    hdrs = ["tmpl_tf_op.h"],
    deps = [
        ":tmpl_op",
        "//machina/core:framework",
        "//machina/lite/kernels/shim:tf_op_shim",
    ],
)

tf_kernel_library(
    name = "tmpl_tflite_op",
    srcs = ["tmpl_tflite_op.cc"],
    hdrs = ["tmpl_tflite_op.h"],
    deps = [
        ":tmpl_op",
        "//machina/lite:mutable_op_resolver",
        "//machina/lite/c:common",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels/shim:op_kernel",
        "//machina/lite/kernels/shim:tflite_op_shim",
        "//machina/lite/kernels/shim:tflite_op_wrapper",
    ],
)

tf_cc_test(
    name = "tmpl_tf_op_test",
    srcs = ["tmpl_tf_op_test.cc"],
    tags = [
        "nochromiumos_arm",
        # Exclude from mobile builds
        "tflite_not_portable",
        "tflite_not_portable_android",
    ],
    deps = [
        ":tmpl_tf_op",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/framework:fake_input",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/kernels:ops_testutil",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

tf_cc_test(
    name = "tmpl_tflite_op_test",
    srcs = ["tmpl_tflite_op_test.cc"],
    tags = [
        "nochromiumos_arm",
        # Exclude from mobile builds
        "tflite_not_portable",
        "tflite_not_portable_android",
    ],
    deps = [
        ":tmpl_tflite_op",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:testlib",
        "//machina/core/kernels:ops_testutil",
        "//machina/lite/c:c_api_types",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

tflite_portable_test_suite()
