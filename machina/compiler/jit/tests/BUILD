load("//machina:machina.bzl", "tf_cc_test")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

licenses(["notice"])

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:private"],
)

cc_library(
    name = "auto_clustering_test_helper",
    testonly = True,
    srcs = ["auto_clustering_test_helper.cc"],
    hdrs = ["auto_clustering_test_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/jit:compilation_passes",
        "//machina/compiler/jit:jit_compilation_passes",
        "//machina/compiler/jit:xla_cluster_util",
        "//machina/compiler/jit:xla_cpu_jit",
        "//machina/compiler/jit:xla_gpu_jit",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/tools/optimization:optimization_pass_runner_lib",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:status_macros",
    ],
)

tf_cc_test(
    name = "auto_clustering_test",
    srcs = ["auto_clustering_test.cc"],
    data = [
        "keras_imagenet_main.golden_summary",
        "keras_imagenet_main.pbtxt",
        "keras_imagenet_main_graph_mode.golden_summary",
        "keras_imagenet_main_graph_mode.pbtxt",
        "opens2s_gnmt_mixed_precision.golden_summary",
        "opens2s_gnmt_mixed_precision.pbtxt.gz",
    ],
    tags = ["config-cuda-only"],
    deps = [
        ":auto_clustering_test_helper",
        "//machina/core:test",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "device_compiler_test_helper",
    testonly = True,
    srcs = ["device_compiler_test_helper.cc"],
    hdrs = ["device_compiler_test_helper.h"],
    visibility = [
        "//machina/compiler/jit:__pkg__",
    ],
    deps = [
        "//machina/compiler/jit:xla_activity_listener",
        "//machina/compiler/jit:xla_compilation_cache_proto_cc",
        "//machina/compiler/jit:xla_cpu_jit",
        "//machina/compiler/jit:xla_gpu_device",
        "//machina/compiler/jit:xla_gpu_jit",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:direct_session",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:ops",
        "//machina/core:test",
        "//machina/core/platform:path",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/service:hlo_proto_cc",
    ],
)

tf_cc_test(
    name = "device_compiler_serialize_test",
    srcs = [
        "device_compiler_serialize_test.cc",
    ],
    tags = [
        "config-cuda-only",
        "no_oss",  # This test only runs with GPU.
        "requires-gpu-nvidia",
        "xla",
    ],
    deps = [
        ":device_compiler_test_helper",
        "//machina/compiler/jit:compilation_passes",
        "//machina/compiler/jit:flags",
        "//machina/core:test",
    ],
)

tf_cc_test(
    name = "device_compiler_serialize_options_test",
    srcs = [
        "device_compiler_serialize_options_test.cc",
    ],
    tags = [
        "config-cuda-only",
        "no_oss",  # This test only runs with GPU.
        "requires-gpu-nvidia",
        "xla",
    ],
    deps = [
        ":device_compiler_test_helper",
        "//machina/compiler/jit:compilation_passes",
        "//machina/compiler/jit:flags",
        "//machina/core:test",
        "//machina/core/framework:graph_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)
