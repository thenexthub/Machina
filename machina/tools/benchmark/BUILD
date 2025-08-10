# Description:
#   Benchmark utility that can run on desktop and Android.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_copts",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "benchmark_model_lib",
    testonly = 1,
    srcs = [
        "benchmark_model.cc",
    ],
    hdrs = [
        "benchmark_model.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = select({
        "//machina:android": [
            "//machina/core:portable_machina_lib",
            "//machina/core:portable_machina_test_lib",
        ],
        "//conditions:default": [
            "//machina/core:core_cpu",
            "//machina/core:framework",
            "//machina/core:framework_internal",
            "//machina/core:framework_lite",
            "//machina/core:lib",
            "//machina/core:protos_all_cc",
            "//machina/core:machina",
            "//machina/core:test",
        ],
    }) + [
        "//machina/core/platform:numbers",
        "//machina/core/util:stats_calculator_portable",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:status",
    ],
)

tf_cc_test(
    name = "benchmark_model_test",
    size = "medium",
    srcs = ["benchmark_model_test.cc"],
    deps = [
        ":benchmark_model_lib",
        "//machina/cc:cc_ops",
        "//machina/cc:scope",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

# This binary may be built for either desktop or Android.
# A typical Android build command will look like the following:
# bazel build machina/core:portable_machina_lib \
# --crosstool_top=//external:android/crosstool \
# --cpu=armeabi-v7a \
# --host_crosstool_top=@bazel_tools//tools/cpp:toolchain
# --config monolithic
tf_cc_binary(
    name = "benchmark_model",
    testonly = 1,
    srcs = ["benchmark_model_main.cc"],
    copts = tf_copts(),
    linkopts = select({
        "//machina:android": [
            "-pie",
            "-s",
            "-landroid",
            "-latomic",
            "-ljnigraphics",
            "-llog",
            "-lm",
            "-z defs",
            "-s",
            "-Wl,--exclude-libs,ALL",  # Exclude syms in all libs from auto export
        ],
        "//conditions:default": [],
    }),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [":benchmark_model_lib"],
)
