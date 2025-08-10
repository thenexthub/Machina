load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# ctc support classes imported directly from TensorFlow.
cc_library(
    name = "ctc_utils",
    hdrs = [
        "ctc_beam_entry.h",
        "ctc_beam_scorer.h",
        "ctc_beam_search.h",
        "ctc_decoder.h",
        "ctc_loss_util.h",
    ],
    deps = [
        ":top_n",
        "//machina/lite/kernels/internal:compatibility",
        "@eigen_archive//:eigen3",
    ],
)

# top_n support classes imported directly from TensorFlow.
cc_library(
    name = "top_n",
    hdrs = [
        "top_n.h",
    ],
    deps = [
        "//machina/lite/kernels/internal:compatibility",
    ],
)

cc_library(
    name = "ctc_beam_search_decoder_op",
    srcs = [
        "ctc_beam_search_decoder.cc",
    ],
    # Suppress warnings that are introduced by Eigen Tensor.
    copts = tflite_copts() + [
        "-Wno-error=reorder",
    ] + select({
        "//machina:ios": ["-Wno-error=invalid-partial-specialization"],
        "//conditions:default": [
        ],
    }),
    deps = [
        ":ctc_utils",
        "//machina/lite:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels:op_macros",
        "//machina/lite/kernels/internal:optimized_base",
        "//machina/lite/kernels/internal:tensor",
        "@flatbuffers",
    ],
)

cc_test(
    name = "ctc_beam_search_decoder_test",
    size = "small",
    srcs = ["ctc_beam_search_decoder_test.cc"],
    tags = ["tflite_not_portable_ios"],
    deps = [
        ":ctc_beam_search_decoder_op",
        "//machina/lite:framework",
        "//machina/lite/core:framework",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)
