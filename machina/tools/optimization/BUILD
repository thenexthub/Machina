# Description:
#   Utilities that perform useful transformations on graphs

load(
    "//machina:machina.bzl",
    "tf_cc_binary",
    "tf_cuda_library",
)
load("//machina:machina.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_cuda_library(
    name = "optimization_pass_runner_lib",
    srcs = ["optimization_pass_runner.cc"],
    hdrs = ["optimization_pass_runner.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:framework_lite",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

tf_cc_binary(
    name = "gpu_optimization_pass_runner",
    srcs = ["gpu_optimization_pass_runner_main.cc"],
    deps = [
        ":optimization_pass_runner_lib",
        "//machina/compiler/jit:xla_cpu_jit",
        "//machina/compiler/jit:xla_gpu_jit",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:core_cpu_base",
        "//machina/core:framework_internal",
        "//machina/core:framework_lite",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:status",
    ],
)
