# Description:
#   JNI-based Java inference interface for TensorFlow.

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "if_android",
    "tf_cc_binary",  # @unused
    "tf_copts",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "jni/version_script.lds",
])

filegroup(
    name = "android_machina_inference_jni_srcs",
    srcs = glob([
        "**/*.cc",
        "**/*.h",
    ]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "android_machina_inference_jni",
    srcs = if_android([":android_machina_inference_jni_srcs"]),
    copts = tf_copts(),
    features = ["-layering_check"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:portable_machina_lib_lite",
        "//machina/java/src/main/native",
    ],
    alwayslink = 1,
)

# JAR with Java bindings to TF.
android_library(
    name = "android_machina_inference_java",
    srcs = glob(["java/**/*.java"]) + ["//machina/java:java_sources"],
    javacopts = [
    ],
    tags = [
        "manual",
        "notap",
    ],
)

# Build the native .so.
# bazel build //machina/tools/android/inference_interface:libmachina_inference.so \
#   --crosstool_top=//external:android/crosstool \
#   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
#   --cpu=armeabi-v7a
LINKER_SCRIPT = "//machina/tools/android/inference_interface:jni/version_script.lds"

# This fails to buiild if converted to tf_cc_binary.
cc_binary(
    name = "libmachina_inference.so",
    copts = tf_copts() + [
        "-ffunction-sections",
        "-fdata-sections",
    ],
    linkopts = if_android([
        "-landroid",
        "-latomic",
        "-ldl",
        "-llog",
        "-lm",
        "-z defs",
        "-s",
        "-Wl,--gc-sections",
        "-Wl,--version-script,$(location {})".format(LINKER_SCRIPT),
    ]),
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":android_machina_inference_jni",
        "//machina/core:portable_machina_lib",
        LINKER_SCRIPT,
    ],
)

cc_library(
    name = "android_machina_inference_native",
    srcs = if_android([":libmachina_inference.so"]),
)
