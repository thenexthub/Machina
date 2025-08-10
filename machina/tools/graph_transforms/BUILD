# Description:
#   Utilities that perform useful transformations on graphs

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:strict.default.bzl", "py_strict_library")
load(
    "//machina:machina.bzl",
    "if_not_windows",
    "tf_cc_binary",
    "tf_cc_test",
    "tf_copts",
)
load("//machina:machina.default.bzl", "get_compatible_with_portable", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "transform_utils",
    srcs = [
        "transform_utils.cc",
    ],
    hdrs = [
        "transform_utils.h",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "transform_utils_test",
    size = "small",
    srcs = ["transform_utils_test.cc"],
    deps = [
        ":transform_utils",
        "//machina/cc:cc_ops",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

cc_library(
    name = "file_utils",
    srcs = [
        "file_utils.cc",
    ],
    hdrs = [
        "file_utils.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "file_utils_test",
    size = "small",
    srcs = ["file_utils_test.cc"],
    deps = [
        ":file_utils",
        "//machina/cc:cc_ops",
        "//machina/core:core_cpu",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

cc_library(
    name = "transforms_lib",
    srcs = [
        "add_default_attributes.cc",
        "backports.cc",
        "flatten_atrous.cc",
        "fold_batch_norms.cc",
        "fold_constants_lib.cc",
        "fold_old_batch_norms.cc",
        "freeze_requantization_ranges.cc",
        "fuse_convolutions.cc",
        "inline_partitionedcall.cc",
        "insert_logging.cc",
        "obfuscate_names.cc",
        "quantize_nodes.cc",
        "quantize_weights.cc",
        "remove_attribute.cc",
        "remove_control_dependencies.cc",
        "remove_device.cc",
        "remove_nodes.cc",
        "rename_attribute.cc",
        "rename_node.cc",
        "rename_op.cc",
        "round_weights.cc",
        "set_device.cc",
        "sort_by_execution_order.cc",
        "sparsify_gather.cc",
        "strip_unused_nodes.cc",
    ],
    hdrs = [
        "fold_constants_lib.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":transform_utils",
        "//machina/c:checkpoint_reader",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core/kernels:quantization_utils",
        "//machina/core/util/tensor_bundle",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ] + if_not_windows([
        "//machina/core:sparse_ops_op_lib",
        "//machina/core:parsing_ops_op_lib",
        "//machina/core:sendrecv_ops_op_lib",
        "//machina/core:io_ops_op_lib",
        "//machina/core:logging_ops_op_lib",
        "//machina/core:lookup_ops_op_lib",
        "//machina/core:data_flow_ops_op_lib",
        "//machina/core:no_op_op_lib",
        "//machina/core:state_ops_op_lib",
        "//machina/core:user_ops_op_lib",
        "//machina/core:training_ops_op_lib",
        "//machina/core:string_ops_op_lib",
        "//machina/core:random_ops_op_lib",
        "//machina/core:rnn_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core:math_ops_op_lib",
        "//machina/core:manip_ops_op_lib",
        "//machina/core:list_ops_op_lib",
        "//machina/core:functional_ops_op_lib",
        "//machina/core:control_flow_ops_op_lib",
        "//machina/core:candidate_sampling_ops_op_lib",
        "//machina/core:array_ops_op_lib",
    ]),
    alwayslink = 1,
)

tf_cc_test(
    name = "transforms_test",
    size = "small",
    srcs = [
        "add_default_attributes_test.cc",
        "backports_test.cc",
        "flatten_atrous_test.cc",
        "fold_batch_norms_test.cc",
        "fold_constants_test.cc",
        "fold_old_batch_norms_test.cc",
        "freeze_requantization_ranges_test.cc",
        "fuse_convolutions_test.cc",
        "inline_partitionedcall_test.cc",
        "insert_logging_test.cc",
        "obfuscate_names_test.cc",
        "quantize_nodes_test.cc",
        "quantize_weights_test.cc",
        "remove_attribute_test.cc",
        "remove_device_test.cc",
        "remove_nodes_test.cc",
        "rename_attribute_test.cc",
        "rename_node_test.cc",
        "rename_op_test.cc",
        "round_weights_test.cc",
        "set_device_test.cc",
        "sort_by_execution_order_test.cc",
        "sparsify_gather_test.cc",
        "strip_unused_nodes_test.cc",
    ],
    deps = [
        ":transform_utils",
        ":transforms_lib",
        "//machina/cc:cc_ops",
        "//machina/cc:sendrecv_ops",
        "//machina/core:bitwise_ops_op_lib",
        "//machina/core:core_cpu",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/kernels:quantization_utils",
        "//machina/core/kernels:quantized_ops",
        "//machina/core/util/tensor_bundle",
    ],
)

filegroup(
    name = "transform_graph_hdrs",
    srcs = [
        "transform_graph.h",
        "transform_utils.h",
    ],
    visibility = ["//machina/python/util:__pkg__"],
)

cc_library(
    name = "transform_graph_lib",
    srcs = ["transform_graph.cc"],
    hdrs = ["transform_graph.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":file_utils",
        ":transform_utils",
        ":transforms_lib",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
    ],
    alwayslink = 1,
)

# This library includes a main function, to make it easy to create other
# versions of the tool linked against different operator libs.
cc_library(
    name = "transform_graph_main_lib",
    srcs = ["transform_graph_main.cc"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":transform_graph_lib",
        ":transforms_lib",
        "//machina/core:framework_internal",
        "//machina/core:lib",
    ],
)

tf_cc_binary(
    name = "transform_graph",
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":transform_graph_main_lib",
    ],
)

tf_cc_test(
    name = "transform_graph_test",
    size = "medium",
    srcs = ["transform_graph_test.cc"],
    deps = [
        ":transform_graph_lib",
        ":transform_utils",
        "//machina/cc:cc_ops",
        "//machina/cc:sendrecv_ops",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

# This library includes a main function, to make it easy to create other
# versions of the tool linked against different operator libs.
cc_library(
    name = "summarize_graph_main_lib",
    srcs = ["summarize_graph_main.cc"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":file_utils",
        ":transform_utils",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ],
)

tf_cc_binary(
    name = "summarize_graph",
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":summarize_graph_main_lib",
    ],
)

tf_cc_binary(
    name = "compare_graphs",
    srcs = ["compare_graphs.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":file_utils",
        ":transform_utils",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework_internal",
        "//machina/core:lib",
    ],
)

py_strict_library(
    name = "transform_graph_py",
    srcs = ["__init__.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/util:_pywrap_transform_graph",
        "//machina/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "transform_graph_py_test",
    size = "small",
    srcs = ["python/transform_graph_test.py"],
    main = "python/transform_graph_test.py",
    tags = ["v1only"],
    deps = [
        ":transform_graph_py",
        "//machina/core:protos_all_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_util",
        "//machina/python/platform:client_testlib",
    ],
)
