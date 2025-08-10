# Placeholder: load py_proto_library
load(
    "//machina:machina.bzl",
    "tf_cc_test",
    "tf_cuda_library",
)
load("//machina:machina.default.bzl", "filegroup")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_proto_library",
    "tf_protos_grappler",
    "tf_pyclif_proto_library",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

alias(
    name = "graph_properties_testdata",
    actual = "//machina/core/grappler/costs/graph_properties_testdata:graph_properties_testdata",
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "op_performance_data",
    srcs = ["op_performance_data.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "//machina/core/framework:attr_value_proto",
        "//machina/core/framework:resource_handle_proto",
        "//machina/core/framework:tensor_proto",
        "//machina/core/framework:tensor_shape_proto",
        "//machina/core/framework:types_proto",
        "//machina/core/protobuf:for_core_protos",
    ],
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "op_performance_data_pyclif",
    proto_lib = ":op_performance_data",
    proto_srcfile = "op_performance_data.proto",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "analytical_cost_estimator.h",
        "cost_estimator.h",
        "graph_memory.h",
        "graph_properties.h",
        "measuring_cost_estimator.h",
        "op_context.h",
        "op_level_cost_estimator.h",
        "utils.h",
        "virtual_placer.h",
        "virtual_scheduler.h",
    ],
    visibility = [
        "//machina/python/grappler:__pkg__",
    ],
)

cc_library(
    name = "graph_properties",
    srcs = ["graph_properties.cc"],
    hdrs = ["graph_properties.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:mutable_graph_view",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/clusters:cluster",
        "//machina/core/grappler/optimizers:evaluation_utils",
        "//machina/core/grappler/utils:functions",
        "//machina/core/grappler/utils:topological_sort",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/types:optional",
    ] + tf_protos_grappler(),
)

tf_cc_test(
    name = "graph_properties_test",
    srcs = ["graph_properties_test.cc"],
    args = ["--heap_check="],  # The GPU tracer leaks memory. TODO(b/185483595): use a dependency instead of a flag
    data = [":graph_properties_testdata"],
    tags = [
        "nomsan",  # TODO(b/160921160): broken by NOAUTOROLLBACK CL
    ],
    deps = [
        ":graph_properties",
        "//machina/cc:cc_ops",
        "//machina/cc:functional_ops",
        "//machina/cc:scope",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:tensor_testutil",
        "//machina/core/graph:mkl_graph_util",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/clusters:single_machine",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//machina/core/grappler/inputs:utils",
    ],
)

cc_library(
    name = "graph_memory",
    srcs = ["graph_memory.cc"],
    hdrs = ["graph_memory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_estimator",
        ":graph_properties",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/clusters:cluster",
        "//machina/core/grappler/clusters:virtual_cluster",
    ],
)

tf_cc_test(
    name = "graph_memory_test",
    srcs = ["graph_memory_test.cc"],
    args = ["--heap_check="],  # The GPU tracer leaks memory. TODO(b/185483595): use a dependency instead of a flag
    deps = [
        ":graph_memory",
        "//machina/cc:cc_ops",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "robust_stats",
    srcs = ["robust_stats.cc"],
    hdrs = ["robust_stats.h"],
    visibility = ["//visibility:public"],
)

tf_cc_test(
    name = "robust_stats_test",
    srcs = ["robust_stats_test.cc"],
    deps = [
        ":robust_stats",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

tf_cuda_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_estimator",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime/gpu:gpu_id",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/clusters:utils",
        "//machina/core/util:overflow",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@eigen_archive//:eigen3",
    ] + tf_protos_grappler(),
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//machina/cc:cc_ops",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/framework:tensor_testutil",
    ],
)

cc_library(
    name = "cost_estimator",
    srcs = ["cost_estimator.cc"],
    hdrs = ["cost_estimator.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "cost_estimator_test",
    srcs = ["cost_estimator_test.cc"],
    deps = [
        ":cost_estimator",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "virtual_placer",
    srcs = ["virtual_placer.cc"],
    hdrs = ["virtual_placer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:devices",
        "//machina/core/grappler/clusters:cluster",
    ],
)

tf_cc_test(
    name = "virtual_placer_test",
    srcs = ["virtual_placer_test.cc"],
    deps = [
        ":virtual_placer",
        "//machina/core:core_cpu",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/grappler/clusters:virtual_cluster",
    ],
)

cc_library(
    name = "op_context",
    hdrs = ["op_context.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
    ] + tf_protos_grappler(),
)

cc_library(
    name = "virtual_scheduler",
    srcs = ["virtual_scheduler.cc"],
    hdrs = ["virtual_scheduler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_estimator",
        ":graph_properties",
        ":op_context",
        ":utils",
        ":virtual_placer",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/grappler/clusters:utils",
        "//machina/core/grappler/utils:transitive_fanin",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "virtual_scheduler_test",
    srcs = ["virtual_scheduler_test.cc"],
    deps = [
        ":graph_properties",
        ":utils",
        ":virtual_placer",
        ":virtual_scheduler",
        "//machina/cc:cc_ops",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/grappler/clusters:virtual_cluster",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "measuring_cost_estimator",
    srcs = ["measuring_cost_estimator.cc"],
    hdrs = ["measuring_cost_estimator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":robust_stats",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/clusters:cluster",
        "//machina/core/grappler/costs:cost_estimator",
        "//machina/core/kernels:ops_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "op_level_cost_estimator",
    srcs = ["op_level_cost_estimator.cc"],
    hdrs = ["op_level_cost_estimator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_estimator",
        ":op_context",
        ":utils",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler/clusters:utils",
        "//machina/core/util:overflow",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:statusor",
    ] + tf_protos_grappler(),
)

tf_cc_test(
    name = "op_level_cost_estimator_test",
    srcs = ["op_level_cost_estimator_test.cc"],
    tags = [
        "no_oss",
        "not_run:arm",
    ],
    deps = [
        ":op_level_cost_estimator",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:status_matchers",
    ],
)

cc_library(
    name = "analytical_cost_estimator",
    srcs = ["analytical_cost_estimator.cc"],
    hdrs = ["analytical_cost_estimator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_estimator",
        ":graph_properties",
        ":op_level_cost_estimator",
        ":utils",
        ":virtual_placer",
        ":virtual_scheduler",
        "//machina/core:core_cpu_base",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:op_types",
        "//machina/core/grappler:utils",
        "//machina/core/util:overflow",
    ] + tf_protos_grappler(),
    alwayslink = 1,
)

tf_cc_test(
    name = "analytical_cost_estimator_test",
    srcs = ["analytical_cost_estimator_test.cc"],
    deps = [
        ":analytical_cost_estimator",
        ":virtual_scheduler",
        "//machina/cc:cc_ops",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/grappler/clusters:virtual_cluster",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "op_performance_data_py_pb2",
#     has_services = 0,
#     visibility = ["//visibility:public"],
#     deps = [":op_performance_data"],
# )
# copybara:uncomment_end
