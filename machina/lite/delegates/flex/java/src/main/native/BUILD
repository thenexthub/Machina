# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Flex delegate for using TensorFlow ops with TensorFlow Lite.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(
    srcs = ["flex_delegate_jni.cc"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "native",
    srcs = ["flex_delegate_jni.cc"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//machina/lite/delegates/flex:delegate",
        "//machina/lite/delegates/utils:simple_delegate",
        "//machina/lite/java/jni",
        "//machina/lite/testing:init_machina",
    ],
    alwayslink = 1,
)
