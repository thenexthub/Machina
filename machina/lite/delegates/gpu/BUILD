load("@bazel_skylib//lib:selects.bzl", "selects")
load("@build_bazel_rules_apple//apple:ios.bzl", "ios_static_framework")
load("@build_bazel_rules_apple//apple:macos.bzl", "macos_dylib")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_cc//cc:objc_library.bzl", "objc_library")
load(
    "//machina/core/platform:build_config_root.bzl",
    "tf_gpu_tests_tags",
)
load("//machina/lite:build_def.bzl", "CXX17_BAZEL_ONLY_COPTS", "tflite_pagesize_linkopts")
load("//machina/lite:special_rules.bzl", "tflite_extra_gles_deps", "tflite_portable_test_suite")
load("//machina/lite/delegates/gpu:build_defs.bzl", "gpu_delegate_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "delegate.h",
    "delegate_options.h",
    "metal_delegate.h",
])

_DELEGATE_NO_GL_DEPS = select({
    "//machina:android": [
        ":async_buffers",
        "//machina/lite/core/async/c:task",
        "//machina/lite/core/async/interop/c:attribute_map",
        "//machina/lite/core/async/interop/c:constants",
        "//machina/lite/delegates/gpu/gl:android_sync",
        "//machina/lite/delegates/gpu/gl:egl_environment",
        "//machina/lite/delegates/utils",
        "//machina/lite/delegates/utils:async_type_helpers",
        "//machina/lite/delegates/utils:ret_macros",
        "//machina/lite/delegates/utils:sync_fence",
    ],
    "//conditions:default": [],
}) + [
    ":android_hardware_buffer",
    ":api",
    ":delegate_options",
    ":tflite_profile",
    #"//third_party/GL:EGL_headers",
    #"//third_party/GL:GLES3_headers",
    # go/keep-sorted start
    "@com_google_absl//absl/container:flat_hash_map",
    "@com_google_absl//absl/container:flat_hash_set",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/types:span",
    "//machina/lite/async:backend_async_kernel_interface",
    "//machina/lite/core/async/interop/c:types",
    "//machina/lite/core/c:common",
    "//machina/lite/delegates/gpu/cl:util",
    "//machina/lite/delegates/gpu/common:data_type",
    "//machina/lite/delegates/gpu/common:model",
    "//machina/lite/delegates/gpu/common:model_builder",
    "//machina/lite/delegates/gpu/common:model_builder_helper",
    "//machina/lite/delegates/gpu/common:quantization_util",
    "//machina/lite/delegates/gpu/common:status",
    "//machina/lite/delegates:serialization",
    "//machina/lite/kernels:kernel_util",
    "//machina/lite/profiling/telemetry",
    "//machina/lite/profiling/telemetry/c:telemetry_setting",
    "//machina/lite/profiling/telemetry/c:telemetry_setting_internal",
    "//machina/lite/profiling/telemetry:telemetry_status",
    "//machina/lite:kernel_api",
    "//machina/lite:minimal_logging",
    # go/keep-sorted end
]

config_setting(
    name = "tflite_gpu_binary_release",
    values = {"copt": "-DTFLITE_GPU_BINARY_RELEASE"},
)

config_setting(
    name = "tflite_gpu_extra_gles_deps",
    constraint_values = [
        "//third_party/bazel_platforms/cpu:x86_64",
        "//third_party/bazel_platforms/os:linux",
    ],
    values = {
        "copt": "-DTFLITE_GPU_EXTRA_GLES_DEPS",
    },
)

cc_library(
    name = "gl_delegate",
    srcs = ["gl_delegate.cc"],
    hdrs = ["gl_delegate.h"],
    linkopts = gpu_delegate_linkopts(),
    deps = [
        "//machina/lite:kernel_api",
        "//machina/lite:minimal_logging",
        "//machina/lite/core/c:common",
        "//machina/lite/delegates/gpu/common:convert",
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:model_builder",
        "//machina/lite/delegates/gpu/common:model_transformer",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:tensor",
        "//machina/lite/delegates/gpu/common/transformations:model_transformations",
        "//machina/lite/delegates/gpu/gl:api",
        "//machina/lite/delegates/gpu/gl:command_queue",
        "//machina/lite/delegates/gpu/gl:compiler",
        "//machina/lite/delegates/gpu/gl:compiler_options",
        "//machina/lite/delegates/gpu/gl:egl_environment",
        "//machina/lite/delegates/gpu/gl:gl_buffer",
        "//machina/lite/delegates/gpu/gl:gl_call",
        "//machina/lite/delegates/gpu/gl:object",
        "//machina/lite/delegates/gpu/gl:object_manager",
        "//machina/lite/delegates/gpu/gl:request_gpu_info",
        "//machina/lite/delegates/gpu/gl:runtime_options",
        "//machina/lite/delegates/gpu/gl/converters:bhwc_to_phwc4",
        "//machina/lite/delegates/gpu/gl/converters:phwc4_to_bhwc",
        "//machina/lite/delegates/gpu/gl/kernels:registry",
        "//machina/lite/delegates/gpu/gl/workgroups:best_effort_calculator",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/types:span",
    ] + select({
        "//conditions:default": [
            "//machina/lite/delegates/gpu/gl:common_cc_fbs",
            "//machina/lite/delegates/gpu/gl:metadata_cc_fbs",
            "//machina/lite/delegates/gpu/gl:workgroups_cc_fbs",
            "//machina/lite/schema:schema_fbs",
            "@flatbuffers",
        ],
        ":tflite_gpu_binary_release": [],
    }) + tflite_extra_gles_deps(),
)

objc_library(
    name = "metal_delegate",
    srcs = ["metal_delegate.mm"],
    hdrs = ["metal_delegate.h"],
    copts = CXX17_BAZEL_ONLY_COPTS,
    module_name = "TensorFlowLiteCMetal",
    sdk_frameworks = ["Metal"],
    deps = [
        "//machina/lite:kernel_api",
        "//machina/lite:minimal_logging",
        "//machina/lite/core/c:common",
        "//machina/lite/delegates/gpu/common:convert",
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:model_builder",
        "//machina/lite/delegates/gpu/common:model_transformer",
        "//machina/lite/delegates/gpu/common:precision",
        "//machina/lite/delegates/gpu/common:quantization_util",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:types",
        "//machina/lite/delegates/gpu/metal:buffer_convert",
        "//machina/lite/delegates/gpu/metal:common",
        "//machina/lite/delegates/gpu/metal:inference_context",
        "//machina/lite/delegates/gpu/metal:metal_spatial_tensor",
        "//machina/lite/kernels:kernel_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
    ],
)

objc_library(
    name = "metal_delegate_internal",
    hdrs = ["metal_delegate_internal.h"],
    copts = CXX17_BAZEL_ONLY_COPTS,
    sdk_frameworks = ["Metal"],
    deps = ["//machina/lite/delegates/gpu:metal_delegate"],
)

# build -c opt --config android_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --linkopt -s --strip always :libmachinalite_gpu_gl.so
cc_binary(
    name = "libmachinalite_gpu_gl.so",
    linkopts = [
        "-Wl,-soname=libmachinalite_gpu_gl.so",
    ] + gpu_delegate_linkopts() + select({
        "//machina:windows": [],
        "//conditions:default": [
            "-fvisibility=hidden",
        ],
    }) + tflite_pagesize_linkopts(),
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [":gl_delegate"],
)

# build -c opt --config android_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --linkopt -s --strip always :libmachinalite_gpu_delegate.so
cc_binary(
    name = "libmachinalite_gpu_delegate.so",
    linkopts = [
        "-Wl,-soname=libmachinalite_gpu_delegate.so",
    ] + gpu_delegate_linkopts() + select({
        "//machina:windows": [],
        "//conditions:default": [
            "-fvisibility=hidden",
        ],
    }),
    linkshared = 1,
    linkstatic = 1,
    tags = [
        "nobuilder",
        "notap",
    ],
    deps = [":delegate"],
)

# bazel build -c opt --cpu ios_arm64 --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always --cxxopt=-std=c++14 :libmachinalite_gpu_metal --apple_platform_type=ios
ios_static_framework(
    name = "machina_lite_gpu_framework",
    hdrs = [
        "metal_delegate.h",
        "metal_delegate_internal.h",
    ],
    minimum_os_version = "12.0",
    deps = [":metal_delegate"],
)

# Note: Support for MacOS is best-effort at the moment.
# bazel build -c opt --copt -Os --copt -DTFLITE_GPU_BINARY_RELEASE --copt -fvisibility=hidden --linkopt -s --strip always --cxxopt=-std=c++14 :machina_lite_gpu_dylib --apple_platform_type=macos
macos_dylib(
    name = "machina_lite_gpu_dylib",
    linkopts = [
        "-all_load",
        "-dead_strip",
    ],
    minimum_os_version = "12.0",
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        ":metal_delegate",
        ":metal_delegate_internal",
    ],
)

cc_library(
    name = "api",
    srcs = ["api.cc"],
    hdrs = ["api.h"],
    deps = [
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/common:status",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/gl:portable",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@opencl_headers",
        "@vulkan_headers//:vulkan_headers_no_prototypes",
    ],
)

cc_library(
    name = "spi",
    hdrs = ["spi.h"],
    deps = [
        ":api",
        "//machina/lite/delegates/gpu/common:access_type",
        "//machina/lite/delegates/gpu/common:status",
    ],
)

# Currently the GPU delegate needs to be built on Android (due to EGL dependency),
# or built with -DCL_DELEGATE_NO_GL (disabling OpenGL backend fallback), or both.
selects.config_setting_group(
    name = "supports_gpu_delegate",
    match_any = [
        "//machina:android",
        "//machina/lite/delegates/gpu/cl:opencl_delegate_no_gl",
    ],
)

cc_library(
    name = "delegate_options",
    srcs = ["delegate_options.cc"],
    hdrs = ["delegate_options.h"],
    deps = ["//machina/lite/core/c:common"],
)

# copybara:uncomment_begin(google-only)
# cc_library(
#     name = "delegate_no_gl",
#     srcs = [
#         # copybara:comment_begin(oss-only)
# "android_version.cc",
#         # copybara:comment_end
#         "delegate.cc",
#     ],
#     hdrs = ["delegate.h"],
#     defines = ["CL_DELEGATE_NO_GL"],
#     linkopts = gpu_delegate_linkopts(),
#     deps = _DELEGATE_NO_GL_DEPS + [
#         "//machina/lite/delegates/gpu/cl:api_no_gl",
#         "//machina/lite/delegates/gpu/gl:api2",
#     ],
# )
# copybara:uncomment_end

cc_library(
    name = "delegate",
    srcs = [
        # copybara:comment_begin(oss-only)
        "android_version.cc",
        # copybara:comment_end
        "delegate.cc",
    ],
    hdrs = ["delegate.h"],
    linkopts = gpu_delegate_linkopts(),
    deps = select({
        "//machina/lite/delegates/gpu/cl:opencl_delegate_no_gl": [],
        "//conditions:default": [
            "//machina/lite/delegates/gpu/gl:api2",
        ],
    }) + _DELEGATE_NO_GL_DEPS + ["//machina/lite/delegates/gpu/cl:api"],
)

cc_library(
    name = "tflite_profile",
    srcs = ["tflite_profile.cc"],
    hdrs = ["tflite_profile.h"],
    deps = [
        "//machina/lite/core/api",
        "//machina/lite/delegates/gpu/common/task:profiling_info",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "android_hardware_buffer",
    srcs = ["android_hardware_buffer.cc"],
    hdrs = ["android_hardware_buffer.h"],
)

cc_test(
    name = "android_hardware_buffer_test",
    srcs = ["android_hardware_buffer_test.cc"],
    deps = [
        ":android_hardware_buffer",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "async_buffers",
    srcs = ["async_buffers.cc"],
    hdrs = ["async_buffers.h"],
    deps = [
        ":android_hardware_buffer",
        ":api",
        "//machina/lite/delegates/gpu/common:data_type",
        "//machina/lite/delegates/gpu/gl:gl_errors",
        "@com_google_absl//absl/status",
    ],
)

cc_test(
    name = "async_buffers_test",
    srcs = ["async_buffers_test.cc"],
    tags = tf_gpu_tests_tags() + [
        "local",
        "nobuilder",
        "notap",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":android_hardware_buffer",
        ":async_buffers",
        ":delegate",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
