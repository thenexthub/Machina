load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:strict.default.bzl", "py_strict_binary", "py_strict_library")
load("//machina:machina.bzl", "if_not_windows")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "get_compatible_with_portable", "tf_py_strict_test", "tf_pybind_cc_library_wrapper", "tf_python_pybind_extension")
load("//machina/core/platform:build_config.bzl", "tf_protos_grappler")
load("//machina/core/platform:build_config_root.bzl", "if_pywrap")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

# TODO(gunan): Investigate making this action hermetic so we do not need
# to run it locally.
cc_library(
    name = "cost_analyzer_lib",
    srcs = ["cost_analyzer.cc"],
    hdrs = ["cost_analyzer.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/clusters:cluster",
        "//machina/core/grappler/costs:analytical_cost_estimator",
        "//machina/core/grappler/costs:cost_estimator",
        "//machina/core/grappler/costs:measuring_cost_estimator",
        "//machina/core/grappler/costs:utils",
    ] + tf_protos_grappler(),
    alwayslink = 1,
)

# Necessary for the pywrap inclusion below. Combining targets does not work
# properly.
tf_pybind_cc_library_wrapper(
    name = "cost_analyzer_headers",
    deps = [
        ":cost_analyzer_lib",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_cost_analyzer",
    srcs = ["cost_analyzer_wrapper.cc"],
    hdrs = [
        "cost_analyzer.h",
        "//machina/cc:pywrap_required_hdrs",
        "//machina/core/grappler:pywrap_required_hdrs",
        "//machina/core/grappler/clusters:pywrap_required_hdrs",
        "//machina/core/grappler/costs:pywrap_required_hdrs",
        "//machina/core/public:session.h",
        "//machina/core/public:session_options.h",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_cost_analyzer.pyi",
    ],
    starlark_only = True,
    deps = [
        ":cost_analyzer_headers",
        "//machina/core:framework_headers_lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_headers_lib",
        "//machina/core/common_runtime/gpu:gpu_id",
        "//machina/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

cc_library(
    name = "model_analyzer_lib",
    srcs = ["model_analyzer.cc"],
    hdrs = ["model_analyzer.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler/costs:graph_properties",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_model_analyzer",
    srcs = ["model_analyzer_wrapper.cc"],
    hdrs = [
        "model_analyzer.h",
        "//machina/core/grappler:pywrap_required_hdrs",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_model_analyzer.pyi",
    ],
    starlark_only = True,
    deps = [
        "//machina/core:framework_headers_lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:protos_all_cc",
        "//machina/python/lib/core:pybind11_status",
        "@pybind11",
    ] + if_pywrap(["//machina/python/grappler:model_analyzer_lib"]),
)

py_strict_library(
    name = "tf_item",
    srcs = ["item.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tf_item",
        "//machina/core:protos_all_py",
        "//machina/core/grappler/costs:op_performance_data_py",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tf_item",
    srcs = ["item_wrapper.cc"],
    hdrs = [
        "//machina/cc:pywrap_required_hdrs",
        "//machina/core/grappler:pywrap_required_hdrs",
        "//machina/core/grappler/clusters:pywrap_required_hdrs",
        "//machina/core/grappler/costs:pywrap_required_hdrs",
        "//machina/core/grappler/utils:pywrap_required_hdrs",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tf_item.pyi",
    ],
    deps = [
        "//machina/core:framework_headers_lib",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_headers_lib",
        "//machina/core/common_runtime/gpu:gpu_id",
        "//machina/python/lib/core:pybind11_status",
        "@pybind11",
    ] + if_not_windows(["//machina/core/grappler/costs:graph_properties"]),  # b/148556093,
)

tf_py_strict_test(
    name = "item_test",
    size = "small",
    srcs = ["item_test.py"],
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
    deps = [
        ":tf_item",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "datasets_test",
    size = "medium",
    srcs = ["datasets_test.py"],
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
    deps = [
        ":tf_item",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:iterator_ops",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "tf_cluster",
    srcs = ["cluster.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tf_cluster",
        "//machina/core:protos_all_py",
        "//machina/core/grappler/costs:op_performance_data_py",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tf_cluster",
    srcs = ["cluster_wrapper.cc"],
    hdrs = [
        "//machina/cc:pywrap_required_hdrs",
    ] + if_pywrap(
        if_false = [
            "//machina/core/grappler:pywrap_required_hdrs",
            "//machina/core/grappler/clusters:pywrap_required_hdrs",
            "//machina/core/grappler/costs:pywrap_required_hdrs",
            "//machina/core/grappler/utils:pywrap_required_hdrs",
        ],
    ),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tf_cluster.pyi",
    ],
    deps = [
        "//machina/core:framework_headers_lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_headers_lib",
        "//machina/core/common_runtime/gpu:gpu_id",
        "//machina/python/lib/core:pybind11_status",
        "@com_google_absl//absl/types:span",
        "@pybind11",
    ] + if_pywrap(
        if_true = [
            "//machina/core/grappler/costs:measuring_cost_estimator",
            "//machina/core/grappler/clusters:single_machine",
        ],
    ),
)

cuda_py_strict_test(
    name = "cluster_test",
    size = "small",
    srcs = ["cluster_test.py"],
    shard_count = 5,
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
        "no_windows",  # b/173520599
        "notap",  # TODO(b/135924227): Re-enable after fixing flakiness.
    ],
    # This test will not run on XLA because it primarily tests the TF Classic flow.
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_cluster",
        ":tf_item",
        "//machina/core:protos_all_py",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "tf_optimizer",
    srcs = ["tf_optimizer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tf_optimizer",
        ":tf_cluster",
        "//machina/core:protos_all_py",
        "@absl_py//absl/logging",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tf_optimizer",
    srcs = ["tf_optimizer_wrapper.cc"],
    hdrs = if_pywrap(
        if_false = [
            "//machina/cc:pywrap_required_hdrs",
            "//machina/core/grappler:pywrap_required_hdrs",
            "//machina/core/grappler/clusters:pywrap_required_hdrs",
            "//machina/core/grappler/costs:pywrap_required_hdrs",
            "//machina/core/grappler/optimizers:pywrap_required_hdrs",
            "//machina/core/grappler/verifiers:pywrap_required_hdrs",
        ],
    ),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tf_optimizer.pyi",
    ],
    # This fails Windows builds. Please check b/266870200 for details.
    #    dynamic_deps = ["//machina/python:_pywrap_machina_internal.so"] + select({
    #        "//machina:macos": ["//machina:libmachina_framework.%s.dylib" % VERSION],
    #        "//conditions:default": ["//machina:libmachina_framework.so.%s" % VERSION],
    #        "//machina:windows": [],
    #    }),
    #    static_deps = tf_python_pybind_static_deps(),
    deps = [
        "//machina/core:framework_headers_lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_headers_lib",
        "//machina/core/common_runtime/gpu:gpu_id",
        "//machina/python/lib/core:pybind11_status",
        "@pybind11",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ] + if_pywrap(
        if_true = [
            "//machina/core/grappler/clusters:cluster",
            "//machina/core/grappler/clusters:utils",
            "//machina/core/grappler:grappler_item_builder",
            "//machina/core/grappler/optimizers:meta_optimizer",
            "//machina/core/grappler/optimizers:graph_optimizer",
            "//machina/core/grappler/verifiers:graph_verifier",
        ],
    ),
)

tf_py_strict_test(
    name = "tf_optimizer_test",
    size = "small",
    srcs = ["tf_optimizer_test.py"],
    tags = [
        "grappler",
        "no_pip",  # tf_optimizer is not available in pip.
    ],
    deps = [
        ":tf_item",
        ":tf_optimizer",
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "memory_optimizer_test",
    size = "medium",
    srcs = ["memory_optimizer_test.py"],
    tags = [
        "grappler",
    ],
    deps = [
        ":tf_optimizer",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:training_lib",
    ],
)

cuda_py_strict_test(
    name = "constant_folding_test",
    size = "medium",
    srcs = ["constant_folding_test.py"],
    tags = [
        "grappler",
    ],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:functional_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "arithmetic_optimizer_test",
    size = "small",
    srcs = ["arithmetic_optimizer_test.py"],
    tags = [
        "grappler",
    ],
    xla_enable_strict_auto_jit = False,
    deps = [
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

# TODO(b/131764887) Remove once LayoutOptimizer is swapped out with GenericLayoutOptimizer.
#
# cuda_py_test(
#     name = "layout_optimizer_test",
#     size = "medium",
#     srcs = [
#         "layout_optimizer_test.py",
#     ],
#     deps = [
#         "//machina/python/platform:client_testlib",
#         "//machina/python/framework:for_generated_wrappers",
#         "//machina/python/ops:array_ops",
#         "//machina/python/ops:functional_ops",
#         "//machina/python/ops:math_ops",
#         "//machina/python:nn",
#         "//machina/python/user_ops:ops",
#         "//machina/python/ops:random_ops",
#         "//machina/python/ops:state_ops",
#         ":tf_cluster",
#         ":tf_optimizer",
#         "//machina/python/training:training",
#         "//third_party/py/numpy",
#         "//machina/core:protos_all_py",
#         "//machina/python/framework:constant_op",
#         "//machina/python/framework:dtypes",
#     ],
#     shard_count = 10,
#     tags = [
#         "grappler",
#     ],
#     # This test will not run on XLA because it primarily tests the TF Classic flow.
#     xla_enable_strict_auto_jit = False,
# )

py_strict_library(
    name = "cost_analyzer",
    srcs = ["cost_analyzer.py"],
    deps = [
        ":_pywrap_cost_analyzer",
        ":tf_cluster",
        ":tf_item",
    ],
)

py_strict_binary(
    name = "cost_analyzer_tool",
    srcs = ["cost_analyzer_tool.py"],
    deps = [
        ":cost_analyzer",
        ":tf_optimizer",
        "//machina/core:protos_all_py",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/platform:gfile",
        "//machina/python/training:saver",
        "@absl_py//absl:app",
    ],
)

tf_py_strict_test(
    name = "cost_analyzer_test",
    size = "small",
    srcs = ["cost_analyzer_test.py"],
    tags = [
        "grappler",
        "no_cuda_on_cpu_tap",
        "no_mac",
        "no_pip",
        "no_windows",  # TODO(b/151942037)
    ],
    deps = [
        ":cost_analyzer",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:adam",
    ],
)

py_strict_library(
    name = "model_analyzer",
    srcs = [
        "model_analyzer.py",
    ],
    deps = [":_pywrap_model_analyzer"],
)

tf_py_strict_test(
    name = "model_analyzer_test",
    size = "small",
    srcs = ["model_analyzer_test.py"],
    tags = [
        "grappler",
        "no_pip",
    ],
    deps = [
        ":model_analyzer",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "auto_mixed_precision_test",
    size = "medium",
    srcs = [
        "auto_mixed_precision_test.py",
    ],
    tags = ["grappler"],
    # This test analyzes the graph, but XLA changes the names of nodes.
    xla_enable_strict_auto_jit = False,
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python:tf2",
        "//machina/python/client:session",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:function",
        "//machina/python/framework:ops",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:nn_impl",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/ops/losses",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:sysconfig",
        "//machina/python/training:adam",
        "//machina/python/training:gradient_descent",
        "//machina/python/util:_pywrap_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "remapper_test",
    size = "medium",
    srcs = ["remapper_test.py"],
    tags = ["grappler"],
    # This test analyzes the graph, but XLA changes the names of nodes.
    xla_enable_strict_auto_jit = False,
    deps = [
        ":tf_optimizer",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:sysconfig",
        "//machina/python/util:_pywrap_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_graph_analyzer",
    srcs = ["graph_analyzer_tool_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_graph_analyzer.pyi",
    ],
    deps = [
        "//machina/core/grappler/graph_analyzer:graph_analyzer_tool",
        "@pybind11",
    ],
)

py_strict_binary(
    name = "graph_analyzer",
    srcs = ["graph_analyzer.py"],
    deps = [
        ":_pywrap_graph_analyzer",
        "@absl_py//absl:app",
    ],
)
