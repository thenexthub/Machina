# TensorFlow ops for audio front-end processing.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:strict.default.bzl", "py_strict_library")
load(
    "//machina:machina.bzl",
    "tf_copts",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_opts_nortti_if_android",
)
load("//machina:machina.default.bzl", "tf_custom_op_py_strict_library", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "audio_microfrontend",
    srcs = ["audio_microfrontend.cc"],
    hdrs = ["audio_microfrontend.h"],
    deps = [
        "//machina/lite:framework",
        "//machina/lite/c:c_api_types",
        "//machina/lite/c:common",
        "//machina/lite/experimental/microfrontend/lib:frontend",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/kernels/internal:reference",
        "@flatbuffers",
    ],
)

cc_library(
    name = "audio_microfrontend_op_lib",
    srcs = ["ops/audio_microfrontend_op.cc"],
    copts = tf_copts(android_optimization_level_override = None) + tf_opts_nortti_if_android() + [
        "-Wno-narrowing",
        "-Wno-sign-compare",
        "-Wno-overloaded-virtual",
    ] + select({
        "//machina:android": [
            # Selective registration uses constexprs with recursive
            # string comparisons; that can lead to compiler errors, so
            # we increase the constexpr recursion depth.
            "-fconstexpr-depth=1024",
            "-Oz",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//machina/lite/experimental/microfrontend/lib:frontend",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:framework",
            "//machina/core:lib",
        ],
    }),
    alwayslink = 1,
)

cc_test(
    name = "audio_microfrontend_test",
    size = "small",
    srcs = ["audio_microfrontend_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":audio_microfrontend",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

tf_custom_op_library(
    name = "python/ops/_audio_microfrontend_op.so",
    srcs = [
        "ops/audio_microfrontend_op.cc",
    ],
    deps = [
        "//machina/lite/experimental/microfrontend/lib:frontend",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["audio_microfrontend_op"],
    deps = [
        "//machina/core:lib",
        "//machina/lite/experimental/microfrontend/lib:frontend",
    ],
)

tf_gen_op_wrapper_py(
    name = "audio_microfrontend_op",
    extra_py_deps = [
        "//machina/python:pywrap_tfe",
        "//machina/python/util:dispatch",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    deps = [":audio_microfrontend_op_op_lib"],
)

tf_custom_op_py_strict_library(
    name = "audio_microfrontend_py",
    srcs = [
        "python/ops/audio_microfrontend_op.py",
    ],
    dso = [":python/ops/_audio_microfrontend_op.so"],
    kernels = [
        ":audio_microfrontend_op_op_lib",
    ],
    deps = [
        ":audio_microfrontend_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:load_library",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
    ],
)

tf_py_strict_test(
    name = "audio_microfrontend_op_test",
    size = "small",
    srcs = ["python/kernel_tests/audio_microfrontend_op_test.py"],
    deps = [
        ":audio_microfrontend_py",
        "//machina:machina_py",
        "//machina/python/framework:ops",
    ],
)
