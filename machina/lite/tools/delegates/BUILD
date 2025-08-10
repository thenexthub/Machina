load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//machina/lite/core/shims:cc_library_with_tflite.bzl", "cc_library_with_tflite", "cc_test_with_tflite")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

common_copts = tflite_copts() + tflite_copts_warnings()

cc_library_with_tflite(
    name = "delegate_provider_hdr",
    hdrs = [
        "delegate_provider.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = common_copts,
    tflite_deps = [
        "//machina/lite/c:common",
    ],
    deps = [
        "//machina/lite/tools:command_line_flags",
        "//machina/lite/tools:logging",
        "//machina/lite/tools:tool_params",
    ],
)

cc_library_with_tflite(
    name = "delegate_provider_lib",
    srcs = [
        "delegate_provider.cc",
    ],
    copts = common_copts,
    tflite_deps = [
        ":delegate_provider_hdr",
    ],
)

# A convenient library for all inference execution providers.
cc_library_with_tflite(
    name = "tflite_execution_providers",
    copts = tflite_copts(),
    tflite_deps = [
        ":xnnpack_delegate_provider",
        ":delegate_provider_lib",
    ],
    deps = [
        ":coreml_delegate_provider",
        ":default_execution_provider",
        ":external_delegate_provider",
        ":gpu_delegate_provider",
        ":hexagon_delegate_provider",
        ":nnapi_delegate_provider",
        "//machina/lite/tools/delegates/experimental/stable_delegate:delegate_provider",
    ],
    alwayslink = 1,
)

cc_library_with_tflite(
    name = "default_execution_provider",
    srcs = ["default_execution_provider.cc"],
    copts = tflite_copts(),
    linkstatic = True,
    tflite_deps = [
        ":delegate_provider_hdr",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

cc_library(
    name = "gpu_delegate_provider",
    srcs = ["gpu_delegate_provider.cc"],
    copts = common_copts + select({
        "//machina:ios": [
            "-xobjective-c++",
        ],
        "//machina:macos_arm64": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":delegate_provider_hdr",
        "//machina/lite/tools/evaluation:utils",
    ] + select({
        "//machina/lite/delegates/gpu:supports_gpu_delegate": [
            "//machina/lite/delegates/gpu:delegate",
        ],
        "//conditions:default": [],
    }) + select({
        "//machina:ios": [
            "//machina/lite/delegates/gpu:metal_delegate",
        ],
        "//machina:macos_arm64": [
            "//machina/lite/delegates/gpu:metal_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "nnapi_delegate_provider",
    srcs = ["nnapi_delegate_provider.cc"],
    copts = common_copts,
    deps = [
        ":delegate_provider_hdr",
        "//machina/lite/delegates/nnapi:nnapi_delegate",
        "//machina/lite/nnapi:nnapi_implementation",
        "//machina/lite/nnapi:nnapi_util",
        "//machina/lite/nnapi/sl:nnapi_support_library",
        "//machina/lite/nnapi/sl:nnapi_support_library_headers",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hexagon_delegate_provider",
    srcs = ["hexagon_delegate_provider.cc"],
    copts = common_copts,
    deps = [
        ":delegate_provider_hdr",
        "//machina/lite/tools/evaluation:utils",
    ] + select({
        "//machina:arm_any": [
            "//machina/lite/delegates/hexagon:hexagon_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "coreml_delegate_provider",
    srcs = ["coreml_delegate_provider.cc"],
    copts = common_copts + select({
        "//machina:ios": [
            "-xobjective-c++",
        ],
        "//machina:macos_arm64": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":delegate_provider_hdr",
        "//machina/lite/tools/evaluation:utils",
    ] + select({
        "//machina:ios": [
            "//machina/lite/delegates/coreml:coreml_delegate",
        ],
        "//machina:macos_arm64": [
            "//machina/lite/delegates/coreml:coreml_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library_with_tflite(
    name = "xnnpack_delegate_provider",
    srcs = ["xnnpack_delegate_provider.cc"],
    copts = tflite_copts(),
    linkstatic = True,
    tflite_deps = [
        ":delegate_provider_hdr",
        "//machina/lite/tools/evaluation:utils",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/lite/delegates/xnnpack:xnnpack_delegate",
        "//machina/lite/tools:tool_params",
    ],
    alwayslink = 1,
)

cc_test(
    name = "xnnpack_delegate_provider_test",
    srcs = ["xnnpack_delegate_provider_test.cc"],
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":delegate_provider_hdr",
        ":xnnpack_delegate_provider",
        "//machina/lite/delegates/xnnpack:xnnpack_delegate",
        "//machina/lite/tools:tool_params",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "external_delegate_provider",
    srcs = ["external_delegate_provider.cc"],
    copts = tflite_copts(),
    linkopts = select({
        "//machina:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        ":delegate_provider_hdr",
        "//machina/lite/delegates/external:external_delegate",
    ],
    alwayslink = 1,
)

cc_test_with_tflite(
    name = "delegate_provider_lib_test",
    size = "small",
    srcs = ["delegate_provider_test.cc"],
    copts = tflite_copts(),
    # See details in https://github.com/bazelbuild/bazel/issues/11552 to avoid
    # lazy symbol binding failure on macOS.
    linkstatic = select({
        "//machina:macos": True,
        "//conditions:default": False,
    }),
    tflite_deps = [
        ":default_execution_provider",
        ":delegate_provider_hdr",
        ":delegate_provider_lib",
        ":xnnpack_delegate_provider",
        "//machina/lite/c:test_util",
    ],
    deps = [
        ":nnapi_delegate_provider",
        "//machina/lite/delegates/utils/dummy_delegate:dummy_delegate_provider",
        "//machina/lite/tools:tool_params",
        "@com_google_googletest//:gtest_main",
    ],
)
