load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")
load("@rules_cc//cc:objc_library.bzl", "objc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

objc_library(
    name = "benchmark_lib",
    srcs = ["main.mm"],
    data = glob([
        "models/*.tflite",
    ]),
    sdk_frameworks = [
        "Metal",
        "QuartzCore",
    ],
    deps = [
        "//machina/lite:framework",
        "//machina/lite:kernel_api",
        "//machina/lite/core:framework",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/delegates/gpu/common:gpu_info",
        "//machina/lite/delegates/gpu/common:model",
        "//machina/lite/delegates/gpu/common:model_builder",
        "//machina/lite/delegates/gpu/common:model_transformer",
        "//machina/lite/delegates/gpu/common:precision",
        "//machina/lite/delegates/gpu/common:shape",
        "//machina/lite/delegates/gpu/common:util",
        "//machina/lite/delegates/gpu/common/transformations:model_transformations",
        "//machina/lite/delegates/gpu/metal:inference_context",
        "//machina/lite/kernels:kernel_util",
    ],
)

# Application for local benchmarking in XCode
ios_application(
    name = "iOSBenchmark",
    testonly = 1,
    bundle_id = "com.machina.lite.delegates.gpu.metal.benchmarking",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = ["Info.plist"],
    minimum_os_version = "12.0",
    provisioning_profile = "//machina/lite/delegates/gpu/metal/benchmarking:provisioning_profile.mobileprovision",
    tags = [
        "local",
        "notap",
    ],
    deps = [":benchmark_lib"],
)
