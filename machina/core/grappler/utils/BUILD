load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "filegroup")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_protos_grappler",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "topological_sort.h",
    ],
    visibility = [
        "//machina/python/grappler:__pkg__",
    ],
)

cc_library(
    name = "scc",
    srcs = ["scc.cc"],
    hdrs = ["scc.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/inputs:utils",
    ],
)

tf_cc_test(
    name = "scc_test",
    srcs = ["scc_test.cc"],
    data = [
        "//machina/core/grappler/costs:graph_properties_testdata",
    ],
    deps = [
        ":scc",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/clusters:virtual_cluster",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "topological_sort",
    srcs = ["topological_sort.cc"],
    hdrs = ["topological_sort.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:graph_topology_view",
        "//machina/core/grappler:graph_view",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "topological_sort_test",
    srcs = ["topological_sort_test.cc"],
    deps = [
        ":topological_sort",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "frame",
    srcs = ["frame.cc"],
    hdrs = ["frame.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_view",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:op_types",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "frame_test",
    size = "small",
    srcs = ["frame_test.cc"],
    deps = [
        ":frame",
        ":graph_view",
        "//machina/core:lib",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "traversal",
    srcs = ["traversal.cc"],
    hdrs = ["traversal.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:graph_topology_view",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "traversal_test",
    srcs = ["traversal_test.cc"],
    deps = [
        ":traversal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "grappler_test",
    testonly = 1,
    srcs = [
        "grappler_test.cc",
    ],
    hdrs = ["grappler_test.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/cc:scope",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu",
        "//machina/core:direct_session",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "grappler_test_test",
    size = "small",
    srcs = ["grappler_test_test.cc"],
    deps = [
        ":grappler_test",
        "//machina/cc:cc_ops",
        "//machina/core:core_cpu",
        "//machina/core:direct_session",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/grappler:utils",
    ],
)

cc_library(
    name = "functions",
    srcs = [
        "functions.cc",
    ],
    hdrs = ["functions.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_base_no_ops",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "functions_test",
    srcs = ["functions_test.cc"],
    deps = [
        ":functions",
        "//machina/cc:cc_ops",
        "//machina/core:all_kernels",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "colocation",
    srcs = ["colocation.cc"],
    hdrs = ["colocation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:utils",
    ],
)

tf_cc_test(
    name = "colocation_test",
    size = "small",
    srcs = ["colocation_test.cc"],
    deps = [
        ":colocation",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

cc_library(
    name = "symbolic_shapes",
    srcs = ["symbolic_shapes.cc"],
    hdrs = ["symbolic_shapes.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ] + tf_protos_grappler(),
)

tf_cc_test(
    name = "symbolic_shapes_test",
    srcs = ["symbolic_shapes_test.cc"],
    deps = [
        ":symbolic_shapes",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "tpu",
    srcs = ["tpu.cc"],
    hdrs = ["tpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:protos_all_cc",
    ] + tf_protos_grappler(),
)

tf_cc_test(
    name = "tpu_test",
    srcs = ["tpu_test.cc"],
    deps = [
        ":tpu",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "canonicalizer",
    srcs = ["canonicalizer.cc"],
    hdrs = ["canonicalizer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
    ],
)

tf_cc_test(
    name = "canonicalizer_test",
    size = "small",
    srcs = ["canonicalizer_test.cc"],
    deps = [
        ":canonicalizer",
        "//machina/core:all_kernels",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

cc_library(
    name = "graph_view_internal",
    hdrs = ["graph_view_internal.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "graph_view_internal_test",
    srcs = ["graph_view_internal_test.cc"],
    deps = [
        ":graph_view",
        ":graph_view_internal",
        "//machina/cc:cc_ops",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "graph_view",
    srcs = ["graph_view.cc"],
    hdrs = ["graph_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_view_internal",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "graph_view_test",
    srcs = ["graph_view_test.cc"],
    deps = [
        ":graph_view",
        ":grappler_test",
        "//machina/cc:cc_ops",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "pattern_utils",
    srcs = ["pattern_utils.cc"],
    hdrs = ["pattern_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_view",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "pattern_utils_test",
    srcs = ["pattern_utils_test.cc"],
    deps = [
        ":pattern_utils",
        "//machina/cc:cc_ops",
        "//machina/cc:cc_ops_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

cc_library(
    name = "transitive_fanin",
    srcs = ["transitive_fanin.cc"],
    hdrs = ["transitive_fanin.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:utils",
    ],
)

tf_cc_test(
    name = "transitive_fanin_test",
    srcs = ["transitive_fanin_test.cc"],
    deps = [
        ":transitive_fanin",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/grappler:utils",
    ],
)
