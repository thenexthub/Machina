# Description:
# Java Native Interface (JNI) library intended for implementing the
# TensorFlow Lite Java API using the TensorFlow Lite CC library.

load("//machina/lite:build_def.bzl", "tflite_copts")
load("//machina/lite:special_rules.bzl", "jni_utils_visibility_allowlist")
load("//machina/lite/core/shims:cc_library_with_tflite.bzl", "alias_with_tflite", "cc_library_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library_with_tflite(
    name = "jni_utils",
    srcs = ["jni_utils.cc"],
    hdrs = ["jni_utils.h"],
    tflite_deps = [
        "//machina/lite/c/jni:jni_utils",
    ],
    visibility = jni_utils_visibility_allowlist(),
    deps = [
        "//machina/lite:error_reporter",
        "//machina/lite/core/c:common",
        "//machina/lite/java/jni",
    ],
)

alias_with_tflite(
    name = "native_framework_only",
    actual = ":native_experimental_framework_only",
)

# Native code needed for TF Lite Java API, including experimental APIs;
# but only the framework code, no ops.
cc_library_with_tflite(
    name = "native_experimental_framework_only",
    srcs = [
        ":native_experimental_only_srcs",
    ],
    hdrs = [":native_hdrs"],
    tflite_deps = [
        ":jni_utils",
        ":native_stable_framework_only",
        "//machina/lite:framework_experimental",
    ],
    deps = [
        "//machina/lite/java/jni",
    ],
    alwayslink = 1,
)

filegroup(
    name = "native_experimental_only_srcs",
    srcs = [
        "nativeinterpreterwrapperexperimental_jni.cc",
    ],
)

filegroup(
    name = "native_hdrs",
    srcs = [
        "op_resolver_lazy_delegate_proxy.h",
    ],
)

# Native code needed for TF Lite Java API, excluding experimental APIs;
# but only the framework code, no ops.
filegroup(
    name = "native_stable_srcs",
    srcs = [
        "interpreter_factory_impl_jni.cc",
        "nativeinterpreterwrapper_jni.cc",
        "nativesignaturerunner_jni.cc",
        "op_resolver_lazy_delegate_proxy.cc",
        "tensor_jni.cc",
        "machina_lite_jni.cc",
    ],
)

filegroup(
    name = "native_api_srcs",
    srcs = [
        "machina_lite_jni.cc",
    ],
)

cc_library_with_tflite(
    name = "native_stable_framework_only",
    srcs = [":native_stable_srcs"],
    hdrs = [":native_hdrs"],
    copts = tflite_copts(),
    linkopts = [
        "-lm",
        "-ldl",
    ],
    tflite_deps = [
        ":jni_utils",
        "//machina/lite:create_op_resolver_header",
        "//machina/lite:framework_stable",
        "//machina/lite:signature_runner",
        "//machina/lite/c/jni:jni_utils",
        "//machina/lite/c:c_api_without_op_resolver",
        "//machina/lite/c:common",
        "//machina/lite/acceleration/configuration/c:delegate_plugin",
        "//machina/lite/acceleration/configuration/c:xnnpack_plugin",
        "//machina/lite/tools:verifier_internal",
    ],
    deps = [
        "//machina/lite:minimal_logging",
        "//machina/lite:op_resolver",
        "//machina/lite:schema_fbs_version",
        "//machina/lite:string_util",
        "//machina/lite:util",
        "//machina/lite/core:framework",
        "//machina/lite/core/api:op_resolver_internal",
        "//machina/lite/delegates/xnnpack:xnnpack_delegate_hdrs_only",
        "//machina/lite/java/jni",
    ],
    alwayslink = 1,
)

alias_with_tflite(
    name = "native",
    actual = ":native_experimental",
)

# Native code and ops needed for the TF Lite Java API, including experimental
# features. This includes all ops. If you want a smaller binary, you should use
# tflite_custom_cc_library or tflite_custom_android_library rules.
cc_library_with_tflite(
    name = "native_experimental",
    copts = tflite_copts(),
    tflite_deps = [
        ":native_experimental_framework_only",
        "//machina/lite:create_op_resolver_with_builtin_ops",
        "//machina/lite:framework",
        "//machina/lite/kernels:builtin_ops",
    ],
    deps = [
        "//machina/lite/core/api",
    ],
    alwayslink = 1,
)

# Native code and ops needed for the TF Lite Java API, excluding experimental
# features. This includes all ops. If you want a smaller binary, you should use
# tflite_custom_cc_library or tflite_custom_android_library rules.
cc_library_with_tflite(
    name = "native_stable",
    copts = tflite_copts(),
    tflite_deps = [
        ":native_stable_framework_only",
        "//machina/lite:create_op_resolver_with_builtin_ops",
        "//machina/lite:framework_stable",
        "//machina/lite/kernels:builtin_ops",
    ],
    deps = [
        "//machina/lite/core/api",
        "//machina/lite/kernels:deprecated_backends",
    ],
    alwayslink = 1,
)

exports_files(
    [
        "exported_symbols.lds",
        "version_script.lds",
    ],
)
