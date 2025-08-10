# Description:
# OVIC Benchmarker Java API.

load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@rules_java//java:defs.bzl", "java_binary", "java_library", "java_test")
load("//machina/java:build_defs.bzl", "JAVACOPTS")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Build targets for OVIC classification.
java_test(
    name = "OvicClassifierTest",
    size = "medium",
    srcs = ["src/test/java/org/machina/ovic/OvicClassifierTest.java"],
    data = [
        "//machina/lite/java/ovic/src/testdata:labels.txt",
        "//machina/lite/java/ovic/src/testdata:ovic_testdata",
    ],
    javacopts = JAVACOPTS,
    tags = [
        "no_oss",
    ],
    test_class = "org.machina.ovic.OvicClassifierTest",
    visibility = ["//visibility:public"],
    deps = [
        "//machina/lite/java/ovic:ovicbenchmarkerlib_java",
        "@com_google_truth",
        "@junit",
    ],
)

java_binary(
    name = "ovic_validator",
    srcs = ["src/main/java/org/machina/ovic/OvicValidator.java"],
    data = [
        "//machina/lite/java/ovic/src/testdata:labels.txt",
    ],
    main_class = "org.machina.ovic.OvicValidator",
    tags = ["no_oss"],
    deps = [
        "//machina/lite/java/ovic:ovicbenchmarkerlib_java",
        "//machina/lite/java/ovic:ovicdetectionbenchmarkerlib_java",
    ],
)

android_library(
    name = "ovicbenchmarkerlib",
    srcs = [
        "src/main/java/org/machina/ovic/OvicBenchmarker.java",
        "src/main/java/org/machina/ovic/OvicClassificationResult.java",
        "src/main/java/org/machina/ovic/OvicClassifier.java",
        "src/main/java/org/machina/ovic/OvicClassifierBenchmarker.java",
    ],
    tags = ["no_oss"],
    deps = [
        "//machina/lite/java:machinalite",
        "//machina/lite/java/src/testhelper/java/org/machina/lite:testhelper",
    ],
)

java_library(
    name = "ovicbenchmarkerlib_java",
    srcs = [
        "src/main/java/org/machina/ovic/OvicClassificationResult.java",
        "src/main/java/org/machina/ovic/OvicClassifier.java",
    ],
    javacopts = JAVACOPTS,
    tags = ["no_oss"],
    deps = [
        "//machina/lite/java:libmachinalite_jni.so",
        "//machina/lite/java:machinalite_javalib",
        "//machina/lite/java/src/main/native",
        "//machina/lite/java/src/testhelper/java/org/machina/lite:testhelper_javalib",
    ],
)

# Build targets for OVIC detection.
java_test(
    name = "OvicDetectorTest",
    size = "medium",
    srcs = ["src/test/java/org/machina/ovic/OvicDetectorTest.java"],
    data = [
        "//machina/lite/java/ovic/src/testdata:coco_labels.txt",
        "//machina/lite/java/ovic/src/testdata:ovic_testdata",
    ],
    javacopts = JAVACOPTS,
    tags = [
        "no_oss",
    ],
    test_class = "org.machina.ovic.OvicDetectorTest",
    visibility = ["//visibility:public"],
    deps = [
        "//machina/lite/java/ovic:ovicdetectionbenchmarkerlib_java",
        "@com_google_truth",
        "@junit",
    ],
)

android_library(
    name = "ovicdetectionbenchmarkerlib",
    srcs = [
        "src/main/java/org/machina/ovic/BoundingBox.java",
        "src/main/java/org/machina/ovic/OvicBenchmarker.java",
        "src/main/java/org/machina/ovic/OvicDetectionResult.java",
        "src/main/java/org/machina/ovic/OvicDetector.java",
        "src/main/java/org/machina/ovic/OvicDetectorBenchmarker.java",
    ],
    deps = [
        "//machina/lite/java:machinalite",
        "//machina/lite/java/src/testhelper/java/org/machina/lite:testhelper",
    ],
)

java_library(
    name = "ovicdetectionbenchmarkerlib_java",
    srcs = [
        "src/main/java/org/machina/ovic/BoundingBox.java",
        "src/main/java/org/machina/ovic/OvicDetectionResult.java",
        "src/main/java/org/machina/ovic/OvicDetector.java",
    ],
    javacopts = JAVACOPTS,
    deps = [
        "//machina/lite/java:libmachinalite_jni.so",
        "//machina/lite/java:machinalite_javalib",
        "//machina/lite/java/src/main/native",
        "//machina/lite/java/src/testhelper/java/org/machina/lite:testhelper_javalib",
    ],
)
