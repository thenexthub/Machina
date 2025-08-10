load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "if_google", "tf_cc_binary")

package_group(
    name = "internal",
    packages = [
        "//machina/compiler/mlir/tfrt/tests/tfrt_fallback/...",
        "//machina/core/runtime_fallback/...",
    ] + if_google([
        "//learning/brain/experimental/mlir/tflite/tfmrt/...",
        "//learning/brain/experimental/tfrt/...",
        "//learning/brain/tfrt/...",
        "//learning/brain/mobile/lite/...",
    ]),
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

# This build target contains fallback kernels only. Some of the native TFRT
# ops/kernels (e.g. eigen based matmul) can be expensive to build, but they are
# not needed for fallback testing.
tf_cc_binary(
    name = "tf_bef_executor",
    testonly = True,
    deps = [
        ":bef_executor_lib",
        "@com_google_absl//absl/strings",
        "//machina/core/platform:stream_executor",
        "//machina/core/runtime_fallback/conversion:conversion_alwayslink",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_kernels_alwayslink",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_op_handler",
        # copybara:uncomment "//machina/core/runtime_fallback/test:forwarding_test_kernels",
        # copybara:uncomment "//machina/core/runtime_fallback/test:tfrt_forwarding_kernels_alwayslink",
        "@tf_runtime//:basic_kernels_alwayslink",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:hostcontext_alwayslink",
        "@tf_runtime//:tensor_alwayslink",
        "@tf_runtime//:test_kernels_alwayslink",
        "@tf_runtime//backends/cpu:core_runtime_alwayslink",
        # copybara:uncomment "@tf_runtime//backends/cpu:image_alwayslink",
        # copybara:uncomment "@tf_runtime//backends/cpu:proto_alwayslink",
        "@tf_runtime//backends/cpu:test_ops_alwayslink",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core:all_kernels",
        ],
    }) + if_cuda([]),
)

cc_library(
    name = "bef_executor_flags",
    testonly = True,
    srcs = ["bef_executor_flags.cc"],
    hdrs = ["bef_executor_flags.h"],
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@tf_runtime//:bef_executor_driver",
    ],
)

cc_library(
    name = "bef_executor_lib",
    testonly = True,
    srcs = [
        "tf_bef_executor_main.cc",
    ],
    visibility = [
        ":internal",
    ],
    deps = if_google([
        "//machina/core/runtime_fallback/test:test_tf_opkernels_alwayslink",
        "//machina/core/runtime_fallback/test:test_tfrt_kernels_alwayslink",
        "//third_party/tf_runtime_google:xprof_tracing_sink_alwayslink",
    ]) + [
        ":bef_executor_flags",
        "//machina/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//machina/core/runtime_fallback/util:fallback_test_util",
        "//machina/core/tfrt/utils:thread_pool",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:bef_executor_driver",
        "@tf_runtime//:hostcontext_alwayslink",
        "@tf_runtime//:io_alwayslink",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core/platform:platform_port",
        ],
    }),
)
