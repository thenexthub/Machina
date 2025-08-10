load("//machina:machina.bzl", "tf_cc_test", "tf_cuda_library")
load("//machina:machina.default.bzl", "filegroup")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "devices.h",
        "grappler_item.h",
        "grappler_item_builder.h",
    ],
    visibility = [
        "//machina/python/grappler:__pkg__",
    ],
)

cc_library(
    name = "op_types",
    srcs = ["op_types.cc"],
    hdrs = ["op_types.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":grappler_item",
        ":utils",
        "//machina/cc:cc_ops",
        "//machina/core:all_kernels",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/framework:tensor_testutil",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_library(
    name = "devices",
    srcs = ["devices.cc"],
    hdrs = ["devices.h"],
    cuda_deps = [
        "@local_xla//xla/stream_executor/gpu:gpu_init",
        "//machina/core/platform:stream_executor",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "@com_google_absl//absl/log",
    ],
)

cc_library(
    name = "graph_topology_view",
    srcs = ["graph_topology_view.cc"],
    hdrs = ["graph_topology_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_view",
        "//machina/core:graph",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "graph_topology_view_test",
    srcs = ["graph_topology_view_test.cc"],
    deps = [
        ":graph_topology_view",
        ":graph_view",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "graph_view",
    srcs = ["graph_view.cc"],
    hdrs = ["graph_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "graph_view_test",
    srcs = ["graph_view_test.cc"],
    deps = [
        ":graph_view",
        ":grappler_item",
        "//machina/cc:cc_ops",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "grappler_item",
    srcs = [
        "grappler_item.cc",
    ],
    hdrs = ["grappler_item.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":op_types",
        ":utils",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler/utils:transitive_fanin",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "grappler_item_builder",
    srcs = [
        "grappler_item_builder.cc",
    ],
    hdrs = ["grappler_item_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":grappler_item",
        ":op_types",
        ":utils",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler/inputs:utils",
        "//machina/core/grappler/optimizers:model_pruner",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "grappler_item_test",
    srcs = ["grappler_item_test.cc"],
    deps = [
        ":grappler_item",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

tf_cc_test(
    name = "grappler_item_builder_test",
    srcs = ["grappler_item_builder_test.cc"],
    deps = [
        ":grappler_item_builder",
        "//machina/cc:cc_ops",
        "//machina/cc:functional_ops",
        "//machina/cc:grad_testutil",
        "//machina/cc:gradients",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "mutable_graph_view",
    srcs = [
        "mutable_graph_view.cc",
    ],
    hdrs = ["mutable_graph_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_view",
        ":op_types",
        ":utils",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "mutable_graph_view_test",
    srcs = ["mutable_graph_view_test.cc"],
    deps = [
        ":grappler_item",
        ":mutable_graph_view",
        ":utils",
        "//machina/cc:cc_ops",
        "//machina/core:graph",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
