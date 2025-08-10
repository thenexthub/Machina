# Description:
#   TensorFlow C++ inference example for labeling images.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_cc_binary", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "wav_to_spectrogram_lib",
    srcs = ["wav_to_spectrogram.cc"],
    hdrs = ["wav_to_spectrogram.h"],
    deps = [
        "//machina/cc:cc_ops",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_binary(
    name = "wav_to_spectrogram",
    srcs = ["main.cc"],
    deps = [
        ":wav_to_spectrogram_lib",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "wav_to_spectrogram_test",
    size = "medium",
    srcs = ["wav_to_spectrogram_test.cc"],
    deps = [
        ":wav_to_spectrogram_lib",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)
