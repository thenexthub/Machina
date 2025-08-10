load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("//machina:machina.bzl", "tf_cc_test", "tf_cuda_library")
load("//machina:machina.default.bzl", "filegroup")
load(
    "//machina/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_cuda_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    cuda_deps = [
        "@local_config_cuda//cuda:cudnn_header",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime/gpu:gpu_id",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    linkstatic = if_cuda(1, 0),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":utils",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/common_runtime/gpu:gpu_id",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "cluster.h",
        "single_machine.h",
        "utils.h",
        "virtual_cluster.h",
    ],
    visibility = [
        "//machina/python/grappler:__pkg__",
    ],
)

cc_library(
    name = "cluster",
    srcs = ["cluster.cc"],
    hdrs = [
        "cluster.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:session_options",
        "//machina/core/grappler:grappler_item",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "virtual_cluster",
    srcs = ["virtual_cluster.cc"],
    hdrs = [
        "virtual_cluster.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":utils",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler/costs:analytical_cost_estimator",
        "//machina/core/grappler/costs:op_level_cost_estimator",
        "//machina/core/grappler/costs:virtual_scheduler",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "virtual_cluster_test",
    srcs = ["virtual_cluster_test.cc"],
    deps = [
        ":virtual_cluster",
        "//machina/cc:cc_ops",
        "//machina/cc:scope",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "single_machine",
    srcs = ["single_machine.cc"],
    hdrs = [
        "single_machine.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":utils",
        "//machina/cc:coordinator",
        "//machina/cc:queue_runner",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/common_runtime:core_cpu",
        "//machina/core/common_runtime:core_cpu_lib",
        "//machina/core/common_runtime:direct_session_internal",
        "//machina/core/common_runtime/gpu:gpu_id",
        "//machina/core/framework:cost_graph_proto_cc",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/grappler:utils",
        "//machina/core/kernels:ops_util",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "single_machine_test",
    srcs = ["single_machine_test.cc"],
    args = ["--heap_check="],  # The GPU tracer leaks memory. TODO(b/185483595): use a dependency instead of a flag
    tags = [
        "no_cuda_on_cpu_tap",
        "no_gpu",
        "nomsan",  # TODO(b/160921160): broken by NOAUTOROLLBACK CL
    ],
    deps = [
        ":single_machine",
        "//machina/cc:cc_ops",
        "//machina/cc:resource_variable_ops",
        "//machina/cc:scope",
        "//machina/core:core_cpu_internal",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)
