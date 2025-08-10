# Description:
#   BenchmarkModel Android harness for Firebase Test Lab.

load("@build_bazel_rules_android//android:rules.bzl", "android_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina/lite:build_def.bzl", "tflite_jni_binary")
load("//machina/lite:special_rules.bzl", "tflite_hexagon_nn_skel_libraries")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

android_binary(
    name = "benchmark_model_firebase",
    srcs = glob([
        "src/**/*.java",
    ]),
    custom_package = "org.machina.lite.benchmark.firebase",
    manifest = "AndroidManifest.xml",
    multidex = "native",
    # In some platforms we don't have an Android SDK/NDK and this target
    # can't be built. We need to prevent the build system from trying to
    # use the target in that case.
    tags = ["manual"],
    deps = [
        ":hexagon_libs",
        ":machinalite_benchmark_firebase_native",
    ],
)

tflite_jni_binary(
    name = "libmachinalite_benchmark_firebase.so",
    srcs = glob([
        "jni/**/*.cc",
        "jni/**/*.h",
    ]),
    deps = [
        "//machina/lite/java/jni",
        "//machina/lite/tools/benchmark:benchmark_tflite_model_lib",
    ],
)

cc_library(
    name = "machinalite_benchmark_firebase_native",
    srcs = ["libmachinalite_benchmark_firebase.so"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "hexagon_libs",
    srcs = select({
        "//machina:android_arm64": [
            "//machina/lite/delegates/hexagon/hexagon_nn:libhexagon_interface.so",
        ] + tflite_hexagon_nn_skel_libraries(),
        "//machina:android_arm": [
            "//machina/lite/delegates/hexagon/hexagon_nn:libhexagon_interface.so",
        ] + tflite_hexagon_nn_skel_libraries(),
        "//conditions:default": [],
    }),
    visibility = ["//visibility:private"],
)
