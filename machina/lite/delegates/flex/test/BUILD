load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_static_framework")
load("@rules_java//java:defs.bzl", "java_library", "java_test")
load("//machina/java:build_defs.bzl", "JAVACOPTS")
load("//machina/lite/delegates/flex:build_def.bzl", "tflite_flex_jni_library")

# Following targets are using for testing selective-built flex delegate
# in Java. Please don't use them for other purposes.

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tflite_flex_jni_library(
    name = "test",
    testonly = 1,
    additional_deps = ["//machina/lite/python/testdata:double_op_and_kernels"],
    models = [
        "//machina/lite:testdata/multi_add_flex.bin",
        "//machina/lite:testdata/double_flex.bin",
    ],
)

java_library(
    name = "test_machinalitelib_flex",
    testonly = 1,
    srcs = ["//machina/lite/delegates/flex/java/src/main/java/org/machina/lite/flex:flex_delegate"],
    javacopts = JAVACOPTS,
    visibility = ["//visibility:private"],
    deps = [
        ":libmachinalite_flex_jni.so",
        "//machina/lite/java:machinalitelib",
        "@org_checkerframework_qual",
    ],
)

java_test(
    name = "SelectiveBuiltInterpreterFlexTest",
    size = "small",
    srcs = [
        "//machina/lite/java:portable_flex_tests",
        "//machina/lite/java:portable_test_utils",
    ],
    data = [
        "//machina/lite:testdata/multi_add_flex.bin",
    ],
    javacopts = JAVACOPTS,
    tags = [
        "no_cuda_on_cpu_tap",  # CUDA + flex is not officially supported.
        "no_gpu",  # GPU + flex is not officially supported.
        "no_oss",  # Currently requires --config=monolithic, b/118895218.
    ],
    test_class = "org.machina.lite.InterpreterFlexTest",
    visibility = ["//visibility:private"],
    deps = [
        ":test_machinalitelib_flex",
        "//machina/lite/java:machinalitelib",
        "@com_google_truth",
        "@junit",
    ],
)

java_test(
    name = "SelectiveBuiltInterpreterFlexWithCustomOpsTest",
    size = "small",
    srcs = [
        "//machina/lite/java:portable_flex_with_custom_ops_tests",
        "//machina/lite/java:portable_test_utils",
    ],
    data = [
        "//machina/lite:testdata/double_flex.bin",
    ],
    javacopts = JAVACOPTS,
    tags = [
        "no_cuda_on_cpu_tap",  # CUDA + flex is not officially supported.
        "no_gpu",  # GPU + flex is not officially supported.
        "no_oss",  # Currently requires --config=monolithic, b/118895218.
    ],
    test_class = "org.machina.lite.InterpreterFlexWithCustomOpsTest",
    visibility = ["//visibility:private"],
    deps = [
        ":test_machinalitelib_flex",
        "//machina/lite/java:machinalitelib",
        "@com_google_truth",
        "@junit",
    ],
)

# For build test only.
ios_static_framework(
    name = "TestTensorFlowLiteSelectTfOps_framework",
    testonly = 1,
    avoid_deps = ["//machina/lite/core/c:common"],
    bundle_name = "TestTensorFlowLiteSelectTfOps",
    minimum_os_version = "12.0",
    deps = [
        ":test_flex_delegate",
    ],
)

build_test(
    name = "framework_build_test",
    # build_test targets are not meant to be run with sanitizers, and can also
    # cause problems with coverage testing.
    tags = [
        "noasan",
        "nomsan",
        "notap",  # TODO(b/415812396): Re-enable once the test is fixed.
        "notsan",
        "nozapfhahn",
    ],
    targets = [
        ":TestTensorFlowLiteSelectTfOps_framework",
    ],
)
