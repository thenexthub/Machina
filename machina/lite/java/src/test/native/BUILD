# Description:
# Java Native Interface (JNI) library for testing the TensorFlow Lite Java API.

load(
    "//machina/lite/core/shims:cc_library_with_tflite.bzl",
    "cc_library_with_tflite",
    "jni_binary_with_tflite",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "native_srcs",
    srcs = [
        "interpreter_test_jni.cc",
        "nnapi_delegate_test_jni.cc",
        "supported_features_jni.cc",
        "test_init_jni.cc",
    ],
)

cc_library_with_tflite(
    name = "native",
    testonly = 1,
    srcs = [
        "interpreter_test_jni.cc",
        "nnapi_delegate_test_jni.cc",
        "supported_features_jni.cc",
    ],
    tflite_deps = [
        ":test_init_jni",
        "//machina/lite/delegates/nnapi/java/src/main/native",
        "//machina/lite/java/src/main/native",
        "//machina/lite/java/src/main/native:jni_utils",
        "//machina/lite/java/src/main/native:native_framework_only",
    ],
    deps = [
        "//machina/lite/core/c:common",
        "//machina/lite/delegates/nnapi:nnapi_delegate",
        "//machina/lite/delegates/xnnpack:xnnpack_delegate",
        "//machina/lite/java/jni",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/nnapi:nnapi_lib",
        "//machina/lite/nnapi/sl:nnapi_support_library_headers",
    ],
    alwayslink = 1,
)

cc_library_with_tflite(
    name = "test_init_jni",
    testonly = 1,
    srcs = [
        "test_init_jni.cc",
    ],
    tflite_deps = [
        "//machina/lite/java/src/main/native:jni_utils",
        "//machina/lite/c:test_util",
    ],
    deps = [
        "//machina/lite/java/jni",
    ],
    alwayslink = 1,
)

# Same as "native", but excluding dependencies on experimental features.
cc_library_with_tflite(
    name = "native_stable",
    testonly = 1,
    srcs = [
        "supported_features_jni.cc",
        "test_init_jni.cc",
    ],
    tflite_deps = [
        "//machina/lite/c:test_util",
        "//machina/lite/delegates/nnapi/java/src/main/native",
        "//machina/lite/java/src/main/native:jni_utils",
        "//machina/lite/java/src/main/native:native_stable",
        "//machina/lite/java/src/main/native:native_stable_framework_only",
    ],
    deps = [
        "//machina/lite/core/c:common",
        "//machina/lite/delegates/xnnpack:xnnpack_delegate",
        "//machina/lite/java/jni",
        "//machina/lite/kernels:kernel_util",
    ],
    alwayslink = 1,
)

jni_binary_with_tflite(
    name = "libmachinalite_test_jni.so",
    testonly = 1,
    linkscript = "//machina/lite/java:tflite_version_script.lds",
    tflite_deps = [":native"],
)

jni_binary_with_tflite(
    name = "libmachinalite_stable_test_jni.so",
    testonly = 1,
    linkscript = "//machina/lite/java:tflite_version_script.lds",
    tflite_deps = [":native_stable"],
)

# Dummy native library which doesn't actually contain the TFLite implementation.
jni_binary_with_tflite(
    name = "libmachinalite_jni.so",
    testonly = 1,
)
