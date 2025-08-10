# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Hexagon delegate Java API using the TensorFlow Lite CC library.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina/lite:build_def.bzl", "tflite_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/lite/delegates/hexagon/java:__subpackages__"],
)

licenses(["notice"])

cc_library(
    name = "native",
    srcs = ["hexagon_delegate_jni.cc"],
    copts = tflite_copts(),
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//machina/lite/c:c_api_types",
        "//machina/lite/delegates/hexagon:hexagon_delegate",
        "//machina/lite/java/jni",
    ],
    alwayslink = 1,
)
