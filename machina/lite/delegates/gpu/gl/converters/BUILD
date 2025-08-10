load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//machina/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load("//machina/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "bhwc_to_phwc4",
    srcs = ["bhwc_to_phwc4.cc"],
    hdrs = ["bhwc_to_phwc4.h"],
    deps = [
        ":util",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/gl:command_queue",
        "//machina/lite/delegates/gpu/gl:gl_buffer",
        "//machina/lite/delegates/gpu/gl:gl_program",
        "//machina/lite/delegates/gpu/gl:gl_shader",
        "//machina/lite/delegates/gpu/gl:variable",
    ],
)

cc_test(
    name = "bhwc_to_phwc4_test",
    size = "small",
    srcs = ["bhwc_to_phwc4_test.cc"],
    linkopts = [
        "-lEGL",
        "-lGLESv2",
    ],
    tags = tf_gpu_tests_tags() + [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":bhwc_to_phwc4",
        "//machina/lite/delegates/gpu/common:convert",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/gl:egl_environment",
        "//machina/lite/delegates/gpu/gl:gl_buffer",
        "//machina/lite/delegates/gpu/gl:portable",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "phwc4_to_bhwc",
    srcs = ["phwc4_to_bhwc.cc"],
    hdrs = ["phwc4_to_bhwc.h"],
    deps = [
        ":util",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/gl:command_queue",
        "//machina/lite/delegates/gpu/gl:gl_buffer",
        "//machina/lite/delegates/gpu/gl:gl_program",
        "//machina/lite/delegates/gpu/gl:gl_shader",
        "//machina/lite/delegates/gpu/gl:variable",
    ],
)

cc_test(
    name = "phwc4_to_bhwc_test",
    size = "small",
    srcs = ["phwc4_to_bhwc_test.cc"],
    linkopts = [
        "-lEGL",
        "-lGLESv2",
    ],
    tags = tf_gpu_tests_tags() + [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":phwc4_to_bhwc",
        "//machina/lite/delegates/gpu/common:convert",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/gl:egl_environment",
        "//machina/lite/delegates/gpu/gl:gl_buffer",
        "//machina/lite/delegates/gpu/gl:portable",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
