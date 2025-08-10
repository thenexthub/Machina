load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("//machina:machina.default.bzl", "tf_cuda_cc_test")
load("//machina/core/platform:build_config_root.bzl", "tf_cuda_tests_tags")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_cuda_cc_test(
    name = "device_tracer_test",
    size = "small",
    srcs = ["device_tracer_test.cc"],
    args = ["--heap_check="],
    tags = tf_cuda_tests_tags() + [
        "gpu_cupti",
        "nomac",
    ],
    deps = [
        "//machina/cc:cc_ops",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:direct_session",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/common_runtime:direct_session_internal",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:ops_util",
        "//machina/core/profiler/lib:profiler_interface",
        "//machina/core/profiler/lib:profiler_session",
        "//machina/core/profiler/utils:xplane_schema",
        "//machina/core/profiler/utils:xplane_utils",
        "//machina/core/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/backends/profiler/gpu:cuda_test",
        "@local_xla//xla/backends/profiler/gpu:cupti_collector",
        "@local_xla//xla/backends/profiler/gpu:device_tracer",
        "@local_xla//xla/tsl/profiler/utils:tf_xplane_visitor",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cupti_headers",
    ]),
)
