load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load(
    "//machina:machina.bzl",
    "tf_cc_binary",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
    deps = [
        ":input_generator",
        "//machina/lite/core/c:common",
        "//machina/lite/testing:split",
        "//machina/lite/testing:tflite_driver",
    ] + select({
        "//conditions:default": [
            "//machina/core:framework_internal",
            "//machina/core:lib",
        ],
        "//machina:android": [
            "//machina/core:portable_machina_lib",
        ],
    }),
)

tf_cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    data = [
        ":testdata/test_input.csv",
        "//machina/lite:testdata/add.bin",
    ],
    deps = [
        ":util",
        "//machina/lite/testing:tflite_driver",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_binary(
    name = "tflite_kernel_runner",
    srcs = ["tflite_kernel_runner.cc"],
    deps = [
        ":util",
    ],
)

tf_cc_binary(
    name = "generate_diff_report",
    srcs = ["generate_diff_report.cc"],
    deps = [
        ":diff_analyzer",
        "//machina/core:framework_internal",
    ],
)

cc_library(
    name = "input_generator",
    srcs = ["input_generator.cc"],
    hdrs = ["input_generator.h"],
    deps = [
        "//machina/lite:framework",
        "//machina/lite:string",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/testing:join",
        "//machina/lite/testing:split",
    ],
)

cc_test(
    name = "input_generator_test",
    size = "small",
    srcs = ["input_generator_test.cc"],
    data = [
        ":testdata/test_input.csv",
        "//machina/lite:testdata/multi_add.bin",
    ],
    deps = [
        ":input_generator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "diff_analyzer",
    srcs = ["diff_analyzer.cc"],
    hdrs = ["diff_analyzer.h"],
    deps = [
        "//machina/lite:string",
        "//machina/lite/core/c:c_api_types",
        "//machina/lite/core/c:common",
        "//machina/lite/testing:split",
    ],
)

tf_cc_test(
    name = "diff_analyzer_test",
    size = "small",
    srcs = ["diff_analyzer_test.cc"],
    data = [
        ":testdata/test_input.csv",
    ],
    deps = [
        ":diff_analyzer",
        "//machina/core:lib",
        "@com_google_googletest//:gtest_main",
    ],
)
