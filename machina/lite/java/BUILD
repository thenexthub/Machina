# Description:
# TensorFlow Lite Java API.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@build_bazel_rules_android//android:rules.bzl", "android_library")
load("@rules_java//java:defs.bzl", "java_library", "java_test")
load("//machina/java:build_defs.bzl", "JAVACOPTS")
load("//machina/lite:build_def.bzl", "tflite_cc_library_with_c_headers_test", "tflite_jni_linkopts", "tflite_linkopts_no_undefined")
load("//machina/lite/core/shims:cc_library_with_tflite.bzl", "android_library_with_tflite", "cc_library_with_tflite", "java_library_with_tflite", "java_test_with_tflite", "jni_binary_with_tflite")
load("//machina/lite/delegates/flex:build_def.bzl", "tflite_flex_android_library")
load("//machina/lite/java:aar_with_jni.bzl", "aar_with_jni", "aar_without_jni")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "src/testdata/add.bin",
    "src/testdata/add_unknown_dimensions.bin",
    "src/testdata/grace_hopper_224.jpg",
    "src/testdata/int32.bin",
    "src/testdata/int64.bin",
    "src/testdata/invalid_model.bin",
    "src/testdata/mul_add_signature_def.bin",
    "src/testdata/multi_signature_def.bin",
    "src/testdata/quantized.bin",
    "src/testdata/string.bin",
    "src/testdata/string_scalar.bin",
    "src/testdata/tile_with_bool_input.bin",
    "src/testdata/uint8.bin",
    "src/testdata/with_custom_op.lite",
    "AndroidManifest.xml",
    "AndroidManifestApi.xml",
    "AndroidManifestGpu.xml",
    "AndroidManifestGpuApi.xml",
    "proguard.flags",
    "tflite_version_script.lds",
])

#-----------------------------------------------------------------------------
# Filegroup targets.

JAVA_SRCS = glob([
    "src/main/java/org/machina/lite/*.java",
    "src/main/java/org/machina/lite/acceleration/*.java",
    "src/main/java/org/machina/lite/annotations/*.java",
]) + [
    "//machina/lite/delegates/nnapi/java/src/main/java/org/machina/lite/nnapi:nnapi_delegate_api_src",
    "//machina/lite/delegates/nnapi/java/src/main/java/org/machina/lite/nnapi:nnapi_delegate_impl_src",
]

# All Java source files.
filegroup(
    name = "java_srcs",
    srcs = JAVA_SRCS,
)

JAVA_API_SRCS = [
    "src/main/java/org/machina/lite/DataType.java",
    "src/main/java/org/machina/lite/Delegate.java",
    "src/main/java/org/machina/lite/DelegateFactory.java",
    "src/main/java/org/machina/lite/InterpreterApi.java",
    "src/main/java/org/machina/lite/InterpreterFactory.java",
    "src/main/java/org/machina/lite/InterpreterFactoryApi.java",
    "src/main/java/org/machina/lite/RuntimeFlavor.java",
    "src/main/java/org/machina/lite/Tensor.java",
    "src/main/java/org/machina/lite/TensorFlowLite.java",
    "src/main/java/org/machina/lite/annotations/UsedByReflection.java",
    "src/main/java/org/machina/lite/acceleration/ValidatedAccelerationConfig.java",
    "//machina/lite/delegates/nnapi/java/src/main/java/org/machina/lite/nnapi:nnapi_delegate_api_src",
]

# Java source files for API uses.
filegroup(
    name = "java_api_srcs",
    srcs = JAVA_API_SRCS,
)

# Java sources excluding API files.
filegroup(
    name = "java_runtime_srcs",
    srcs = [x for x in JAVA_SRCS if x not in JAVA_API_SRCS],
)

# Java source files for just the stable (non-experimental) APIs.
filegroup(
    name = "java_stable_srcs",
    srcs = [
        "src/main/java/org/machina/lite/DataType.java",
        "src/main/java/org/machina/lite/DataTypeUtils.java",
        "src/main/java/org/machina/lite/Delegate.java",
        "src/main/java/org/machina/lite/DelegateFactory.java",
        "src/main/java/org/machina/lite/InterpreterApi.java",
        "src/main/java/org/machina/lite/InterpreterFactory.java",
        "src/main/java/org/machina/lite/InterpreterFactoryApi.java",
        "src/main/java/org/machina/lite/InterpreterFactoryImpl.java",
        "src/main/java/org/machina/lite/InterpreterImpl.java",
        "src/main/java/org/machina/lite/NativeInterpreterWrapper.java",
        "src/main/java/org/machina/lite/NativeSignatureRunnerWrapper.java",
        "src/main/java/org/machina/lite/RuntimeFlavor.java",
        "src/main/java/org/machina/lite/Tensor.java",
        "src/main/java/org/machina/lite/TensorFlowLite.java",
        "src/main/java/org/machina/lite/TensorImpl.java",
        "src/main/java/org/machina/lite/acceleration/ValidatedAccelerationConfig.java",
        "src/main/java/org/machina/lite/annotations/UsedByReflection.java",
        "src/main/java/org/machina/lite/package-info.java",
        "//machina/lite/delegates/nnapi/java/src/main/java/org/machina/lite/nnapi:nnapi_delegate_src",
    ],
)

# Java source files for just the experimental APIs.
JAVA_EXPERIMENTAL_SRCS = [
    "src/main/java/org/machina/lite/Interpreter.java",
    "src/main/java/org/machina/lite/NativeInterpreterWrapperExperimental.java",
    "src/main/java/org/machina/lite/package-info.java",
]

filegroup(
    name = "java_experimental_srcs",
    srcs = JAVA_EXPERIMENTAL_SRCS,
)

filegroup(
    name = "java_stable_runtime_srcs",
    srcs = [x for x in JAVA_SRCS if x not in JAVA_API_SRCS and x not in JAVA_EXPERIMENTAL_SRCS],
)

# C headers for the Native APIs.

# TODO(b/277186081) Move this list to `../c/BUILD` when aar_with_jni supports filegroups in the headers parameter.
# LINT.IfChange(TFLITE_HEADERS)
TFLITE_HEADERS = [
    # TODO(b/175298345): Clean up and if possible remove c:common.h and core/c:common.h here.
    "//machina/lite:builtin_ops.h",
    "//machina/lite/c:builtin_op_data.h",
    "//machina/lite/c:c_api.h",
    "//machina/lite/c:c_api_experimental.h",
    "//machina/lite/c:c_api_opaque.h",
    "//machina/lite/c:c_api_types.h",
    "//machina/lite/c:common.h",
    "//machina/lite/core/async/c:types.h",
    "//machina/lite/core/c:c_api.h",
    "//machina/lite/core/c:c_api_opaque.h",
    "//machina/lite/core/c:c_api_types.h",
    "//machina/compiler/mlir/lite/core/c:tflite_types.h",
    "//machina/lite/core/c:builtin_op_data.h",
    "//machina/compiler/mlir/lite/core/c:builtin_op_data.h",
    "//machina/lite/core/c:c_api_experimental.h",
    "//machina/lite/core/c:common.h",
    "//machina/lite/core/c:operator.h",
    "//machina/lite/delegates/nnapi:nnapi_delegate_c_api.h",
]
# LINT.ThenChange(../c/BUILD:exported_headers)

filegroup(
    name = "tflite_headers",
    srcs = TFLITE_HEADERS,
)

# LINT.IfChange(tflite_acceleration_exported_headers)
filegroup(
    name = "tflite_acceleration_headers",
    srcs = [
        "//machina/lite/acceleration/configuration/c:delegate_plugin.h",
        "//machina/lite/acceleration/configuration/c:gpu_plugin.h",
        "//machina/lite/acceleration/configuration/c:xnnpack_plugin.h",
        "//machina/lite/core/acceleration/configuration/c:delegate_plugin.h",
        "//machina/lite/core/acceleration/configuration/c:gpu_plugin.h",
        "//machina/lite/core/acceleration/configuration/c:xnnpack_plugin.h",
    ],
)
# LINT.ThenChange(
#     ../acceleration/configuration/c/BUILD:tflite_acceleration_exported_headers,
#     ../core/acceleration/configuration/c/BUILD:tflite_acceleration_exported_headers
# )

#-----------------------------------------------------------------------------
# AAR targets.

# Building machina-lite.aar including 4 variants of .so
# To build an aar for release, run below command:
# bazel build -c opt --fat_apk_cpu=x86,x86_64,arm64-v8a,armeabi-v7a \
# --define=android_dexmerger_tool=d8_dexmerger \
# --define=android_incremental_dexing_tool=d8_dexbuilder \
#   machina/lite/java:machina-lite
aar_with_jni(
    name = "machina-lite",
    android_library = ":machinalite",
    headers = TFLITE_HEADERS,
)

# No JNI should be in this AAR. Only package in the LICENSE.
aar_without_jni(
    name = "machina-lite-api",
    android_library = ":machinalite_api",
)

# Same as "machina-lite" (above), but excluding experimental APIs.
aar_with_jni(
    name = "machina-lite-stable",
    android_library = ":machinalite_stable",
    headers = [
        "//machina/lite:builtin_ops.h",
        "//machina/lite/c:c_api.h",
        "//machina/lite/c:c_api_types.h",
        "//machina/lite/core/async/c:types.h",
        "//machina/lite/core/c:c_api.h",
        "//machina/lite/core/c:c_api_types.h",
        "//machina/lite/core/c:operator.h",
        # TODO(b/175298345): move the stable parts of common.h into
        # a separate header file that contains no experimental APIs
        # (perhaps also replacing concrete struct types with opaque types
        # and accessor functions), and include that header here.
    ],
)

# EXPERIMENTAL: AAR target for using TensorFlow ops with TFLite. Note that this
# .aar contains *only* the Flex delegate for using select tf ops; clients must
# also include the core `machina-lite` runtime.
aar_with_jni(
    name = "machina-lite-select-tf-ops",
    android_library = ":machinalite_flex",
)

# EXPERIMENTAL: AAR target for the GPU acceleration API, EXCLUDING implementation.
# Note that this .aar contains *only* the GPU delegate API; clients must also include a GPU delegate
# implementation, as well as the core `machina-lite` runtime.
aar_without_jni(
    name = "machina-lite-gpu-api",
    android_library = ":machinalite_gpu_api",
)

# EXPERIMENTAL: AAR target for GPU acceleration API and implementation. Note that this .aar contains
# *only* the GPU delegate; clients must also include the core `machina-lite` runtime.
aar_with_jni(
    name = "machina-lite-gpu",
    android_library = ":machinalite_gpu_impl",
    headers = [
        "//machina/lite/delegates/gpu:delegate.h",
        "//machina/lite/delegates/gpu:delegate_options.h",
    ],
)

#-----------------------------------------------------------------------------
# android_library targets.

# 'machinalite' is the public android_library target for the TF Lite Java
# APIs & Runtime, including experimental APIs.  (See also
# 'machinalite_stable' below which includes only the non-experimental APIs;
# if your code does not require the use of experimental APIs, it may be
# preferable to use that one.)
android_library_with_tflite(
    name = "machinalite",
    # Note that we need to explicitly list all the srcs here, rather than
    # including a dependency on ":machinalite_javalib", in order
    # for the class files to get included in the generated AAR file.
    srcs = [":java_runtime_srcs"],
    manifest = "AndroidManifest.xml",
    proguard_specs = ["proguard.flags"],
    tflite_deps = [
        ":machinalite_native",
    ],
    exports = [
        ":machinalite_api",
    ],
    deps = [
        ":machinalite_api",
        "@org_checkerframework_qual",
    ],
)

# 'machinalite-api' is the public android_library target for TF Lite Java
# APIs. Only stable APIs are included. This library corresponds to the public
# Maven package "org.machina:machina-lite-api".
# This target does not include the TF Lite Runtime, which nevertheless is
# required and must be provided via a separate dependency.
android_library(
    name = "machinalite_api",
    srcs = [":java_api_srcs"],
    manifest = "AndroidManifestApi.xml",
    proguard_specs = ["proguard.flags"],
    deps = [
        "@org_checkerframework_qual",
    ],
)

# 'machinalite_stable' is the public android_library target for the
# TF Lite Java APIs & Runtime, excluding experimental APIs.
android_library_with_tflite(
    name = "machinalite_stable",
    # Note that we need to directly includes all the
    # required Java source files directly in "srcs" rather than
    # depending on them via "deps"/"exports"; this is needed when
    # building the AAR file since the current AAR building process
    # doesn't include the transitive Java dependencies.
    srcs = [":java_stable_srcs"],
    manifest = "AndroidManifest.xml",
    proguard_specs = ["proguard.flags"],
    tflite_deps = [
        ":machinalite_native_stable",
    ],
    deps = [
        "@org_checkerframework_qual",
    ],
)

# EXPERIMENTAL: Android target that supports TensorFlow op execution with TFLite.
# Note that this library contains *only* the Flex delegate and its Java wrapper for using
# select TF ops; clients must also include the core `machinalite` runtime.
#
# The library is generated by tflite_flex_android_library rule. This rule can also be used
# to generate trimmed library that only contain kernels for flex ops used in
# a set of models by listing them in the models parameter. Ex:
# tflite_flex_android_library(
#     name = "machinalite_flex",
#     models = [model1, model2],
# )
#
# The tflite_flex_android_library rule also generate the libmachinalite_flex_jni.so as
# an intermediate target.
tflite_flex_android_library(
    name = "machinalite_flex",
    visibility = ["//visibility:public"],
)

# EXPERIMENTAL: Android target for GPU acceleration. Note that this library
# contains *only* the GPU delegate and its Java wrapper; clients must also
# include the core `machinalite` runtime.
# Note that AndroidManifestGpuApi.xml usage requires AGP 4.2.0+.
alias(
    name = "machinalite_gpu",
    actual = "machinalite_gpu_impl",
)

# EXPERIMENTAL: Android target for the implementation of the GPU acceleration API, including the
# native library. Note that this library contains *only* the GPU delegate and its Java wrapper;
# clients must also include the core `machinalite` runtime.
# Note that AndroidManifestGpuApi.xml usage requires AGP 4.2.0+.
android_library(
    name = "machinalite_gpu_impl",
    # Note that we need to directly includes all the Java source files we intend to ship directly in
    # "srcs" rather than depending on them via "deps"/"exports"; this is needed when building the
    # AAR file since the current AAR building process doesn't include the transitive Java
    # dependencies. The API target however can be an export, because it is shipped in a different
    # AAR file.
    srcs = ["//machina/lite/delegates/gpu/java/src/main/java/org/machina/lite/gpu:gpu_delegate_impl"],
    # Note that this uses the standard manifest and doesn't export it: the declaration is required
    # because android_library targets require a non-empty Android package in Bazel. The API target
    # exports the GPU manifest.
    manifest = "AndroidManifestGpu.xml",
    exports = [
        ":machinalite_gpu_api",
        ":machinalite_gpu_native",
    ],
    deps = [
        ":machinalite_gpu_api",
        "//machina/lite/java:machinalite_api",
        "@maven//:androidx_annotation_annotation",
    ],
)

# EXPERIMENTAL: Android target for the implementation of the GPU acceleration API,
# EXCLUDING the native library.
android_library(
    name = "machinalite_gpu_impl_java",
    srcs = ["//machina/lite/delegates/gpu/java/src/main/java/org/machina/lite/gpu:gpu_delegate_impl"],
    deps = [
        ":machinalite_api",
        ":machinalite_gpu_api",
        "@maven//:androidx_annotation_annotation",
    ],
)

# 'machinalite_java' is the public android_library target for the TF Lite Java
# APIs & Runtime, including experimental APIs, but EXCLUDING NATIVE CODE dependencies.
android_library(
    name = "machinalite_java",
    srcs = [":java_srcs"],
    proguard_specs = ["proguard.flags"],
    deps = [
        "@org_checkerframework_qual",
    ],
)

# 'machinalite_java_stable' is the public android_library target for the TF Lite Java
# APIs & Runtime, excluding experimental APIs, and EXCLUDING NATIVE CODE dependencies.
android_library(
    name = "machinalite_java_stable",
    srcs = [":java_stable_srcs"],
    proguard_specs = ["proguard.flags"],
    deps = [
        "@org_checkerframework_qual",
    ],
)

# EXPERIMENTAL: Android target for GPU acceleration API, EXCLUDING implementation.
# Note that this library contains *only* the GPU delegate API; clients must also include
# an implementation, as well as the core `machinalite` runtime.
# Note that AndroidManifestGpuApi.xml usage requires AGP 4.2.0+.
android_library(
    name = "machinalite_gpu_api",
    srcs = ["//machina/lite/delegates/gpu/java/src/main/java/org/machina/lite/gpu:gpu_delegate"],
    exports_manifest = 1,
    manifest = "AndroidManifestGpuApi.xml",
    proguard_specs = ["proguard.flags"],
    deps = [":machinalite_api"],
)

#-----------------------------------------------------------------------------
# java_library targets that include only Java source dependencies,
# not native code.  The corresponding native code is still required
# and must be provided via a separate dependency.

# Java source dependencies for all TF Lite Java APIs, including experimental APIs.
java_library(
    name = "machinalite_javalib",
    javacopts = JAVACOPTS,
    exports = [
        ":machinalite_javalib_experimental",
        ":machinalite_javalib_stable",
    ],
)

# Java source dependencies for TF Lite Java APIs and Runtime, excluding experimental APIs.
java_library(
    name = "machinalite_javalib_stable",
    srcs = [":java_stable_runtime_srcs"],
    javacopts = JAVACOPTS,
    exports = [
        ":machinalite_api_javalib",
    ],
    deps = [
        ":machinalite_api_javalib",
        "@org_checkerframework_qual",
    ],
)

# Java source dependencies for TF Lite Java APIs and Runtime, including experimental APIs.
java_library(
    name = "machinalite_javalib_experimental",
    srcs = [":java_experimental_srcs"],
    javacopts = JAVACOPTS,
    visibility = ["//visibility:private"],
    exports = [
        ":machinalite_api_javalib",
    ],
    deps = [
        ":machinalite_api_javalib",
        ":machinalite_javalib_stable",
        "@org_checkerframework_qual",
    ],
)

# EXPERIMENTAL: Java source dependencies target that supports TensorFlow op execution with TFLite.
java_library(
    name = "machinalite_flex_javalib",
    srcs = ["//machina/lite/delegates/flex/java/src/main/java/org/machina/lite/flex:flex_delegate"],
    javacopts = JAVACOPTS,
    deps = [
        ":machinalite_javalib",
        "@org_checkerframework_qual",
    ],
)

java_library(
    name = "machinalite_api_javalib",
    srcs = [":java_api_srcs"],
    javacopts = JAVACOPTS,
    deps = [
        "@org_checkerframework_qual",
    ],
)

java_library_with_tflite(
    name = "test_init",
    testonly = True,
    srcs = [
        "src/test/java/org/machina/lite/TestInit.java",
    ],
)

#-----------------------------------------------------------------------------
# java_library targets that also include native code dependencies.

java_library_with_tflite(
    name = "machinalitelib",
    srcs = [":java_runtime_srcs"],
    javacopts = JAVACOPTS,
    tflite_jni_binaries = [
        ":libmachinalite_jni.so",
    ],
    exports = [
        ":machinalite_api_javalib",
    ],
    deps = [
        ":machinalite_api_javalib",
        "@org_checkerframework_qual",
    ],
)

java_library_with_tflite(
    name = "machinalitelib_stable",
    srcs = [":java_stable_srcs"],
    javacopts = JAVACOPTS,
    tflite_jni_binaries = [
        ":libmachinalite_jni_stable.so",
    ],
    deps = [
        "@org_checkerframework_qual",
    ],
)

# EXPERIMENTAL: Java target that supports TensorFlow op execution with TFLite.
java_library(
    name = "machinalitelib_flex",
    srcs = ["//machina/lite/delegates/flex/java/src/main/java/org/machina/lite/flex:flex_delegate"],
    javacopts = JAVACOPTS,
    deps = [
        ":libmachinalite_flex_jni.so",  # Generated by tflite_flex_android_library rule.
        ":machinalitelib",
        "@org_checkerframework_qual",
    ],
)

#-----------------------------------------------------------------------------
# Tests.

java_test_with_tflite(
    name = "TensorFlowLiteTest",
    size = "small",
    srcs = [
        "src/test/java/org/machina/lite/TensorFlowLiteTest.java",
    ],
    javacopts = JAVACOPTS,
    # We want to ensure that every test case in the test also verifies that the
    # native libraries get loaded.  But Java doesn't provide a way to unload
    # native libraries: there's no System.unloadLibrary().  So instead, we just
    # run every test case in this test in a separate shard, which ensures that
    # they run in a separate process that doesn't have the native libraries
    # loaded yet.
    # LINT.IfChange(TensorFlowLiteTestShardCount)
    shard_count = 5,  # grep -c @Test src/test/java/org/machina/lite/TensorFlowLiteTest.java
    # LINT.ThenChange(src/test/java/org/machina/lite/TensorFlowLiteTest.java)
    tags = [
        "v1only",
    ],
    test_class = "org.machina.lite.TensorFlowLiteTest",
    tflite_deps = [
        ":test_init",
    ],
    tflite_jni_binaries = [
        "//machina/lite/java/src/test/native:libmachinalite_test_jni.so",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":machinalite_javalib",
        "@com_google_truth",
        "@junit",
    ],
)

java_test(
    name = "TensorFlowLiteNoNativeLibTest",
    size = "small",
    srcs = ["src/test/java/org/machina/lite/TensorFlowLiteNoNativeLibTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.machina.lite.TensorFlowLiteNoNativeLibTest",
    visibility = ["//visibility:private"],
    deps = [
        ":machinalite_javalib",
        "@com_google_truth",
        "@junit",
        "@org_checkerframework_qual",
    ],
)

java_test_with_tflite(
    name = "TensorFlowLiteInvalidNativeLibTest",
    size = "small",
    srcs = ["src/test/java/org/machina/lite/TensorFlowLiteInvalidNativeLibTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.machina.lite.TensorFlowLiteInvalidNativeLibTest",
    tflite_jni_binaries = [
        "//machina/lite/java/src/test/native:libmachinalite_jni.so",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":machinalite_javalib",
        "@com_google_truth",
        "@junit",
        "@org_checkerframework_qual",
    ],
)

java_test_with_tflite(
    name = "DataTypeTest",
    size = "small",
    srcs = ["src/test/java/org/machina/lite/DataTypeTest.java"],
    javacopts = JAVACOPTS,
    test_class = "org.machina.lite.DataTypeTest",
    tflite_deps = [
        ":machinalitelib",
    ],
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_truth",
        "@junit",
    ],
)

java_test_with_tflite(
    name = "NativeInterpreterWrapperTest",
    size = "small",
    srcs = [
        "src/test/java/org/machina/lite/NativeInterpreterWrapperTest.java",
    ],
    data = [
        # The files named as <data_type>.bin reshape the incoming tensor from (2, 8, 8, 3) to (2, 4, 4, 12).
        "src/testdata/add.bin",
        "src/testdata/int32.bin",
        "src/testdata/int64.bin",
        "src/testdata/invalid_model.bin",
        "src/testdata/string.bin",
        # Takes a scalar string and reshapes to a rank-1, single element string.
        "src/testdata/string_scalar.bin",
        "src/testdata/uint8.bin",
        "src/testdata/with_custom_op.lite",
    ],
    javacopts = JAVACOPTS,
    test_class = "org.machina.lite.NativeInterpreterWrapperTest",
    tflite_deps = [
        ":test_init",
    ],
    tflite_jni_binaries = [
        "//machina/lite/java/src/test/native:libmachinalite_test_jni.so",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":machinalite_javalib",
        "@com_google_truth",
        "@junit",
    ],
)

# TODO: generate large models at runtime, instead of storing them.
java_test_with_tflite(
    name = "InterpreterTest",
    size = "small",
    srcs = [
        "src/test/java/org/machina/lite/InterpreterTest.java",
        "src/test/java/org/machina/lite/SupportedFeatures.java",
        "src/test/java/org/machina/lite/TestUtils.java",
    ],
    data = [
        "src/testdata/add.bin",
        "src/testdata/add_unknown_dimensions.bin",
        "src/testdata/mul_add_signature_def.bin",
        "src/testdata/multi_signature_def.bin",
        "src/testdata/tile_with_bool_input.bin",
        "//machina/lite:testdata/dynamic_shapes.bin",
        "//machina/lite:testdata/multi_add.bin",
        "//machina/lite:testdata/multi_add_flex.bin",
    ],
    javacopts = JAVACOPTS,
    test_class = "org.machina.lite.InterpreterTest",
    tflite_deps = [
        ":test_init",
    ],
    tflite_jni_binaries = [
        "//machina/lite/java/src/test/native:libmachinalite_test_jni.so",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":machinalite_javalib",
        "@com_google_truth",
        "@junit",
    ],
)

# Disabled under the (b/279852433) because caused an error in the OSS
# TODO(zhurakovskyi): Uncomment when fixed.
#
# copybara:uncomment_begin
# java_test_with_tflite(
#     name = "InterpreterApiTest",
#     size = "small",
#     srcs = [
#         "src/test/java/org/machina/lite/InterpreterApiTest.java",
#         "src/test/java/org/machina/lite/SupportedFeatures.java",
#         "src/test/java/org/machina/lite/TestUtils.java",
#     ],
#     data = [
#         "src/testdata/add.bin",
#         "src/testdata/add_unknown_dimensions.bin",
#         "src/testdata/mul_add_signature_def.bin",
#         "src/testdata/multi_signature_def.bin",
#         "src/testdata/tile_with_bool_input.bin",
#         "//machina/lite:testdata/dynamic_shapes.bin",
#         "//machina/lite:testdata/multi_add.bin",
#         "//machina/lite:testdata/multi_add_flex.bin",
#     ],
#     javacopts = JAVACOPTS,
#     test_class = "org.machina.lite.InterpreterApiTest",
#     tflite_deps = [
#         ":test_init",
#     ],
#     tflite_jni_binaries = [
#         "//machina/lite/java/src/test/native:libmachinalite_stable_test_jni.so",
#     ],
#     visibility = ["//visibility:private"],
#     deps = [
#         ":machinalite_javalib_stable",
#         "//third_party/java/mockito",
#         "@com_google_truth",
#         "@junit",
#     ],
# )
# copybara:uncomment_end

java_test_with_tflite(
    name = "InterpreterApiNoRuntimeTest",
    size = "small",
    srcs = [
        "src/test/java/org/machina/lite/InterpreterApiNoRuntimeTest.java",
        "src/test/java/org/machina/lite/SupportedFeatures.java",
        "src/test/java/org/machina/lite/TestUtils.java",
    ],
    data = [
        "src/testdata/add.bin",
    ],
    javacopts = JAVACOPTS,
    test_class = "org.machina.lite.InterpreterApiNoRuntimeTest",
    tflite_deps = [
        ":test_init",
    ],
    tflite_jni_binaries = [
        "//machina/lite/java/src/test/native:libmachinalite_stable_test_jni.so",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":machinalite_api_javalib",
        "@com_google_truth",
        "@junit",
    ],
)

# Commented out under the (b/279852433) because caused an error in the OSS
# TODO(zhurakovskyi): Uncomment when fixed.
# copybara:uncomment_begin
# java_test_with_tflite(
#     name = "NnApiDelegateNativeTest",
#     size = "small",
#     srcs = [
#         "src/test/java/org/machina/lite/NnApiDelegateNativeTest.java",
#         "src/test/java/org/machina/lite/SupportedFeatures.java",
#         "src/test/java/org/machina/lite/TestUtils.java",
#     ],
#     data = [
#         "src/testdata/add.bin",
#     ],
#     tags = ["no_mac"],
#     test_class = "org.machina.lite.NnApiDelegateNativeTest",
#     tflite_deps = [
#         ":test_init",
#     ],
#     tflite_jni_binaries = [
#         "//machina/lite/java/src/test/native:libmachinalite_test_jni.so",
#     ],
#     visibility = ["//visibility:private"],
#     deps = [
#         ":machinalite_javalib",
#         "//third_party/java/mockito",
#         "@com_google_truth",
#         "@junit",
#     ],
# )
# copybara:uncomment_end

java_test_with_tflite(
    name = "NnApiDelegateTest",
    size = "small",
    srcs = [
        "src/test/java/org/machina/lite/SupportedFeatures.java",
        "src/test/java/org/machina/lite/TestUtils.java",
        "src/test/java/org/machina/lite/nnapi/NnApiDelegateTest.java",
    ],
    data = [
        "src/testdata/add.bin",
    ],
    javacopts = JAVACOPTS,
    tags = ["no_mac"],
    test_class = "org.machina.lite.nnapi.NnApiDelegateTest",
    tflite_deps = [
        ":test_init",
    ],
    tflite_jni_binaries = [
        "//machina/lite/java/src/test/native:libmachinalite_test_jni.so",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":machinalite_javalib",
        "@com_google_truth",
        "@junit",
    ],
)

java_test(
    name = "InterpreterFlexTest",
    size = "small",
    srcs = [
        "src/test/java/org/machina/lite/InterpreterFlexTest.java",
        "src/test/java/org/machina/lite/TestUtils.java",
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
        ":machinalitelib",
        ":machinalitelib_flex",
        "@com_google_truth",
        "@junit",
    ],
)

java_test_with_tflite(
    name = "TensorTest",
    size = "small",
    srcs = [
        "src/test/java/org/machina/lite/TensorTest.java",
    ],
    data = [
        "src/testdata/add.bin",
        "src/testdata/int32.bin",
        "src/testdata/int64.bin",
        "src/testdata/quantized.bin",
        "src/testdata/string.bin",
    ],
    javacopts = JAVACOPTS,
    test_class = "org.machina.lite.TensorTest",
    tflite_deps = [
        ":test_init",
    ],
    tflite_jni_binaries = [
        "//machina/lite/java/src/test/native:libmachinalite_test_jni.so",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":machinalite_javalib",
        "@com_google_truth",
        "@junit",
    ],
)

java_test(
    name = "InterpreterCustomizedAndroidBuildTest",
    size = "small",
    srcs = [
        "src/test/java/org/machina/lite/InterpreterCustomizedAndroidBuildTest.java",
        "src/test/java/org/machina/lite/TestUtils.java",
    ],
    data = [
        "//machina/lite:testdata/add.bin",
        "//machina/lite:testdata/test_model.bin",
    ],
    javacopts = JAVACOPTS,
    # Add customized libmachinalite_jni.so to java_path
    jvm_flags = ["-Djava.library.path=third_party/machina/lite/testing"],
    tags = [
        "v1only",
    ],
    test_class = "org.machina.lite.InterpreterCustomizedAndroidBuildTest",
    visibility = ["//visibility:private"],
    deps = [
        "//machina/lite/testing:customized_tflite_for_add_ops",
        "@com_google_truth",
        "@junit",
    ],
)

tflite_cc_library_with_c_headers_test(
    name = "native_headers_lib_for_test",
    testonly = True,
    srcs = [],
    hdrs = TFLITE_HEADERS,
    visibility = ["//visibility:private"],
    deps = [],
)

#-----------------------------------------------------------------------------
# Test filegroup targets.

# Java test source files that test the stable API implementation.
filegroup(
    name = "java_stable_test_srcs",
    srcs = [
        "src/test/java/org/machina/lite/DataTypeTest.java",
        "src/test/java/org/machina/lite/InterpreterApiTest.java",
        "src/test/java/org/machina/lite/NativeInterpreterWrapperTest.java",
        "src/test/java/org/machina/lite/SupportedFeatures.java",
        "src/test/java/org/machina/lite/TensorFlowLiteTest.java",
        "src/test/java/org/machina/lite/TensorTest.java",
        "src/test/java/org/machina/lite/TestInit.java",
        "src/test/java/org/machina/lite/TestUtils.java",
    ],
)

# portable_tests includes files for running TFLite interpreter tests.
filegroup(
    name = "portable_tests",
    srcs = [
        "src/test/java/org/machina/lite/InterpreterMobileNetTest.java",
        "src/test/java/org/machina/lite/InterpreterTest.java",
        "src/test/java/org/machina/lite/SupportedFeatures.java",
        "src/test/java/org/machina/lite/TensorFlowLiteTest.java",
        "src/test/java/org/machina/lite/TestInit.java",
        "src/test/java/org/machina/lite/nnapi/NnApiDelegateTest.java",
    ],
    visibility = ["//visibility:public"],
)

# portable_flex_tests includes files for testing interpreter with Flex delegate.
filegroup(
    name = "portable_flex_tests",
    srcs = [
        "src/test/java/org/machina/lite/InterpreterFlexTest.java",
    ],
    visibility = ["//visibility:public"],
)

# portable_flex_with_custom_ops_tests includes files for testing Flex delegate
# with models containing user's defined ops.
filegroup(
    name = "portable_flex_with_custom_ops_tests",
    srcs = [
        "src/test/java/org/machina/lite/InterpreterFlexWithCustomOpsTest.java",
    ],
    visibility = ["//visibility:public"],
)

# portable_test_utils include utilities for loading files and processing images.
filegroup(
    name = "portable_test_utils",
    srcs = [
        "src/test/java/org/machina/lite/TestUtils.java",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "portable_gpu_tests",
    srcs = [
        "src/test/java/org/machina/lite/InterpreterTestHelper.java",
        "src/test/java/org/machina/lite/gpu/CompatibilityListTest.java",
        "src/test/java/org/machina/lite/gpu/GpuDelegateTest.java",
    ],
    visibility = ["//visibility:public"],
)

#-----------------------------------------------------------------------------
# cc_library targets for linking against the .so files for the JNI code.

alias(
    name = "libmachinalite_jni",
    actual = select({
        "//conditions:default": ":libmachinalite_jni.so",
    }),
)

cc_library_with_tflite(
    name = "machinalite_native",
    tflite_jni_binaries = ["libmachinalite_jni.so"],
)

cc_library_with_tflite(
    name = "machinalite_native_stable",
    tflite_jni_binaries = ["libmachinalite_jni_stable.so"],
)

cc_library_with_tflite(
    name = "machinalite_test_native",
    testonly = True,
    tflite_jni_binaries = ["//machina/lite/java/src/test/native:libmachinalite_test_jni.so"],
)

cc_library_with_tflite(
    name = "machinalite_gpu_native",
    tflite_jni_binaries = ["libmachinalite_gpu_jni.so"],
    visibility = ["//visibility:private"],
)

#-----------------------------------------------------------------------------
# JNI binary targets for generating the .so files for the JNI code.

# JNI shared library containing Java API native code, including experimental APIs.
# This library will get automatically loaded by TensorFlowLite.init().
jni_binary_with_tflite(
    name = "libmachinalite_jni.so",
    linkopts = tflite_jni_linkopts() + tflite_linkopts_no_undefined(),
    linkscript = ":tflite_version_script.lds",
    tflite_deps = [
        "//machina/lite/delegates/nnapi/java/src/main/native",
        "//machina/lite/java/src/main/native",
    ],
    deps = [
        # Note that we explicitly include the C API here for convenience, as it
        # allows bundling of the C lib w/ AAR distribution.
        "//machina/lite/c:c_api",
        "//machina/lite/c:c_api_experimental",
    ],
)

# JNI shared library containing Java API native code, excluding experimental APIs.
# This library will get automatically loaded by TensorFlowLite.init(),
# if libmachinalite_jni.so isn't found.
jni_binary_with_tflite(
    name = "libmachinalite_jni_stable.so",
    linkopts = tflite_jni_linkopts() + tflite_linkopts_no_undefined(),
    linkscript = ":tflite_version_script.lds",
    tflite_deps = [
        "//machina/lite/delegates/nnapi/java/src/main/native",
        "//machina/lite/java/src/main/native:native_stable",
    ],
    deps = [
        # Note that we explicitly include the C API here for convenience, as it
        # allows bundling of the C lib w/ AAR distribution.
        "//machina/lite/c:c_api",
    ],
)

# EXPERIMENTAL: Native target that supports GPU acceleration.
# Unlike libmachinalite_jni.so, this will NOT get loaded automatically by
# TensorFlowLite.init(); instead this library should be explicitly loaded by the
# client code, using System.LoadLibrary, before TensorFlowLite.init() is called.
jni_binary_with_tflite(
    name = "libmachinalite_gpu_jni.so",
    linkopts = tflite_jni_linkopts() + tflite_linkopts_no_undefined(),
    linkscript = ":gpu_version_script.lds",
    tflite_deps = [
        "//machina/lite/delegates/gpu/java/src/main/native",
    ],
)

#-----------------------------------------------------------------------------
# bzl_library targets.

bzl_library(
    name = "aar_with_jni_bzl",
    srcs = ["aar_with_jni.bzl"],
    visibility = ["//machina/lite:__subpackages__"],
)
