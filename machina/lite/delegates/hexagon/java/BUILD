load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina/lite:build_def.bzl", "tflite_jni_binary", "tflite_jni_linkopts")
load("//machina/lite/java:aar_with_jni.bzl", "aar_with_jni")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# EXPERIMENTAL: Native target that runs inference on the Hexagon backend.
# The Hexagon backend-related targets are intentionally not included in BUILD.bazel.
tflite_jni_binary(
    name = "libmachinalite_hexagon_jni.so",
    linkopts = tflite_jni_linkopts() + select({
        "//machina:android": ["-ldl"],
        "//conditions:default": [],
    }),
    linkscript = "//machina/lite/delegates/hexagon:version_script.lds",
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//machina/lite/delegates/hexagon/java/src/main/native",
    ],
)

cc_library(
    name = "machinalite_hexagon",
    srcs = [
        "libmachinalite_hexagon_jni.so",
    ] + select({
        # copybara:uncomment_begin(separate repo in OSS)
        # "//machina:android_arm64": ["//machina/lite/delegates/hexagon/hexagon_nn:libhexagon_interface.so"],
        # "//machina:android_arm": ["//machina/lite/delegates/hexagon/hexagon_nn:libhexagon_interface.so"],
        # copybara:uncomment_end_and_comment_begin
        "//machina:android_arm64": ["@hexagon_nn//:hexagon/arm64-v8a/libhexagon_interface.so"],
        "//machina:android_arm": ["@hexagon_nn//:hexagon/armeabi-v7a/libhexagon_interface.so"],
        # copybara:comment_end
        "//conditions:default": [],
    }),
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
)

android_library(
    name = "machinalite_java_hexagon",
    srcs = ["//machina/lite/delegates/hexagon/java/src/main/java/org/machina/lite:hexagon_delegate"],
    manifest = "AndroidManifest.xml",
    proguard_specs = ["proguard.flags"],
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":machinalite_hexagon",
        "//machina/lite/java:machinalite_java",
        "@org_checkerframework_qual",
    ],
)

aar_with_jni(
    name = "machina-lite-hexagon",
    android_library = ":machinalite_java_hexagon",
    headers = ["//machina/lite/delegates/hexagon:hexagon_delegate.h"],
)
