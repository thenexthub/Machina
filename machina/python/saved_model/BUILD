# Description:
# TensorFlow SavedModel.

load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.bzl", "if_google", "tf_cc_test")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test", "tf_pybind_cc_library_wrapper", "tf_python_pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    # TODO(drpng): change that to //third_party/machina:internal
    # when we have migrated all users.
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_library(
    name = "saved_model",
    srcs = ["saved_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":builder",
        ":constants",
        ":fingerprinting",
        ":load",
        ":loader",
        ":main_op",
        ":method_name_updater",
        ":save",
        ":signature_constants",
        ":signature_def_utils",
        ":simple_save",
        ":tag_constants",
        ":utils",
        "//machina/python/saved_model/model_utils",
        "//machina/python/saved_model/registration",
    ],
)

py_strict_library(
    name = "constants",
    srcs = ["constants.py"],
    deps = [
        ":pywrap_saved_model",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "signature_constants",
    srcs = ["signature_constants.py"],
    deps = ["//machina/python/util:tf_export"],
)

py_strict_library(
    name = "tag_constants",
    srcs = ["tag_constants.py"],
    deps = ["//machina/python/util:tf_export"],
)

py_strict_library(
    name = "builder",
    srcs = [
        "builder.py",
        "builder_impl.py",
    ],
    deps = [
        ":fingerprinting_utils",
        ":path_helpers",
        ":pywrap_saved_model",
        ":signature_def_utils",
        "//machina/core:protos_all_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:saver",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
        # copybara:uncomment "//machina/tools/proto_splitter/python:saved_model",
    ],
)

py_strict_library(
    name = "loader",
    srcs = [
        "loader.py",
        "loader_impl.py",
    ],
    deps = [
        ":constants",
        ":path_helpers",
        ":pywrap_saved_model",
        ":signature_def_utils",
        ":utils",
        "//machina/core:protos_all_py",
        "//machina/python/framework:ops",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:saver",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "loader_test",
    size = "small",
    srcs = ["loader_test.py"],
    deps = [
        ":builder",
        ":loader",
        ":signature_def_utils",
        ":utils",
        "//machina/python/client:session",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:resource_variables_toggle",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:saver",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "simple_save",
    srcs = ["simple_save.py"],
    deps = [
        ":builder",
        ":signature_constants",
        ":signature_def_utils",
        ":tag_constants",
        "//machina/python/framework:ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "main_op",
    srcs = [
        "main_op.py",
        "main_op_impl.py",
    ],
    deps = [
        "//machina/python/framework:ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:variables",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "saved_model_test",
    size = "small",
    srcs = ["saved_model_test.py"],
    data = ["//machina/cc/saved_model:saved_model_half_plus_two"],
    tags = ["no_windows"],
    deps = [
        ":builder",
        ":constants",
        ":fingerprinting",
        ":loader",
        ":main_op",
        ":signature_def_utils",
        ":tag_constants",
        ":utils",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:test_ops",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variables_toggle",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/training",
        "//machina/python/training:saver_test_utils",
        "//machina/python/util:compat",
    ],
)

py_strict_library(
    name = "path_helpers",
    srcs = ["path_helpers.py"],
    deps = [
        ":constants",
        "//machina/python/lib/io:file_io",
        "//machina/python/util:compat",
    ],
)

py_strict_library(
    name = "utils",
    srcs = [
        "utils.py",
        "utils_impl.py",
    ],
    deps = [
        ":nested_structure_coder",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/framework:byte_swap_tensor",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.py"],
    deps = [
        ":nested_structure_coder",
        ":utils",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "signature_def_utils",
    srcs = [
        "signature_def_utils.py",
        "signature_def_utils_impl.py",
    ],
    deps = [
        ":signature_constants",
        ":utils",
        "//machina/core:protos_all_py",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_util",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "signature_def_utils_test",
    size = "small",
    srcs = ["signature_def_utils_test.py"],
    deps = [
        ":signature_constants",
        ":signature_def_utils",
        ":utils",
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "simple_save_test",
    size = "small",
    srcs = ["simple_save_test.py"],
    deps = [
        ":loader",
        ":signature_constants",
        ":simple_save",
        ":tag_constants",
        "//machina/python/framework:ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "signature_serialization",
    srcs = ["signature_serialization.py"],
    deps = [
        ":function_serialization",
        ":revived_types",
        ":signature_constants",
        "//machina/python/eager:def_function",
        "//machina/python/eager:function",
        "//machina/python/eager/polymorphic_function:attributes",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/trackable:base",
        "//machina/python/types:core",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "save_context",
    srcs = ["save_context.py"],
)

tf_py_strict_test(
    name = "save_context_test",
    srcs = ["save_context_test.py"],
    deps = [
        ":save_context",
        ":save_options",
        "//machina/python:extra_py_tests_deps",
        "//machina/python/eager:test",
    ],
)

py_strict_library(
    name = "save",
    srcs = ["save.py"],
    deps = [
        ":builder",
        ":fingerprinting_utils",
        ":function_serialization",
        ":path_helpers",
        ":pywrap_saved_model",
        ":revived_types",
        ":save_context",
        ":save_options",
        ":signature_constants",
        ":signature_def_utils",
        ":signature_serialization",
        ":tag_constants",
        ":tracing_utils",
        ":utils",
        "//machina/core:protos_all_py",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint:checkpoint_options",
        "//machina/python/checkpoint:functional_saver",
        "//machina/python/checkpoint:graph_view",
        "//machina/python/checkpoint:save_util_v1",
        "//machina/python/checkpoint:util",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:function",
        "//machina/python/eager/polymorphic_function",
        "//machina/python/eager/polymorphic_function:concrete_function",
        "//machina/python/eager/polymorphic_function:saved_model_exported_concrete",
        "//machina/python/eager/polymorphic_function:saved_model_utils",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:error_interpolation",
        "//machina/python/framework:errors",
        "//machina/python/framework:function",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:versions",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/saved_model/registration",
        "//machina/python/trackable:asset",
        "//machina/python/trackable:base",
        "//machina/python/trackable:resource",
        "//machina/python/trackable:trackable_utils",
        "//machina/python/training/saving:trace_saveable_util",
        "//machina/python/types:core",
        "//machina/python/util:compat",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_stack",
        "@absl_py//absl/logging",
    ] + if_google([
        "//machina/tools/proto_splitter/python:saved_model",
    ]),
)

py_strict_library(
    name = "tracing_utils",
    srcs = [
        "tracing_utils.py",
    ],
    deps = [
        "//machina/python/checkpoint:saveable_compat",
        "//machina/python/checkpoint:tensor_callable",
        "//machina/python/eager:def_function",
        "//machina/python/eager:function",
    ],
)

tf_py_strict_test(
    name = "save_test",
    srcs = ["save_test.py"],
    deps = [
        ":load",
        ":loader",
        ":save",
        ":save_options",
        ":signature_constants",
        ":tag_constants",
        "//machina/core:protos_all_py",
        "//machina/core/config:flags_py",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint/sharding:sharding_policies",
        "//machina/python/client:session",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute:mirrored_strategy",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:remote",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:versions",
        "//machina/python/lib/io:file_io",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_switch_case",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:gfile",
        "//machina/python/trackable:asset",
        "//machina/python/trackable:autotrackable",
        "//machina/python/training:saver",
        "//machina/python/training:server_lib",
        "//machina/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

# copybara:uncomment_begin(google-only)
#
# tf_py_strict_test(
#     name = "proto_splitter_save_test",
#     srcs = ["proto_splitter_save_test.py"],
#     tags = [
#         "no_mac",  # TODO(b/291933687)
#         "no_windows",  # TODO(b/291001524)
#     ],
#     deps = [
#         ":save",
#         ":save_options",
#         "//machina/python/eager:def_function",
#         "//machina/python/eager:test",
#         "//machina/python/framework:constant_op",
#         "//machina/python/module",
#         "//machina/tools/proto_splitter:constants",
#         "//third_party/py/numpy",
#         "@absl_py//absl/testing:parameterized",
#     ],
# )
#
# copybara:uncomment_end

py_strict_library(
    name = "load",
    srcs = ["load.py"],
    tags = [
        "ignore_for_dep=third_party.py.tf_keras.optimizers.optimizer_v2",
    ],
    deps = [
        ":fingerprinting",
        ":fingerprinting_utils",
        ":function_deserialization",
        ":load_options",
        ":load_v1_in_v2",
        ":loader",
        ":path_helpers",
        ":pywrap_saved_model",
        ":revived_types",
        ":utils",
        "//machina/core:protos_all_py",
        "//machina/core/function/capture:restore_captures",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint:checkpoint_options",
        "//machina/python/checkpoint:graph_view",
        "//machina/python/checkpoint:restore",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:distribute_utils",
        "//machina/python/distribute:values_util",
        "//machina/python/eager:context",
        "//machina/python/eager:function",
        "//machina/python/eager/polymorphic_function:saved_model_utils",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/saved_model/registration",
        "//machina/python/trackable:asset",
        "//machina/python/trackable:autotrackable",
        "//machina/python/trackable:base",
        "//machina/python/trackable:data_structures",
        "//machina/python/trackable:resource",
        "//machina/python/trackable:trackable_utils",
        "//machina/python/training:py_checkpoint_reader",
        "//machina/python/training/saving:saveable_object_util",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "load_v1_in_v2",
    srcs = ["load_v1_in_v2.py"],
    deps = [
        ":function_deserialization",
        ":loader",
        ":pywrap_saved_model",
        ":signature_serialization",
        "//machina/python/eager:context",
        "//machina/python/eager:lift_to_graph",
        "//machina/python/eager:wrap_function",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/platform:tf_logging",
        "//machina/python/trackable:asset",
        "//machina/python/trackable:autotrackable",
        "//machina/python/trackable:resource",
        "//machina/python/training:monitored_session",
        "//machina/python/training:saver",
        "//machina/python/util:nest",
    ],
)

cuda_py_strict_test(
    name = "load_test",
    srcs = ["load_test.py"],
    shard_count = 10,
    tags = [
        "no_gpu",  # TODO(b/136560979): flaky
        "no_mac",  # TODO(b/124822121): Re-enable this test.
    ],
    deps = [
        ":load",
        ":load_options",
        ":loader",
        ":save",
        ":save_options",
        ":tag_constants",
        "//machina/core/function/trace_type",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint:saveable_compat",
        "//machina/python/client:session",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:readers",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:function",
        "//machina/python/eager:test",
        "//machina/python/eager:wrap_function",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:function",
        "//machina/python/framework:op_callbacks",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:versions",
        "//machina/python/lib/io:file_io",
        "//machina/python/lib/io:tf_record",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond_v2",
        "//machina/python/ops:custom_gradient",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/trackable:asset",
        "//machina/python/trackable:autotrackable",
        "//machina/python/trackable:resource",
        "//machina/python/training:monitored_session",
        "//machina/python/types:core",
        "//machina/python/util:tf_inspect",
        "@absl_py//absl/testing:parameterized",
    ] + if_google([
        "//machina/cc/experimental/tf2:runtime_pybind",
    ]),
)

tf_py_strict_test(
    name = "load_v1_in_v2_test",
    srcs = ["load_v1_in_v2_test.py"],
    shard_count = 2,
    deps = [
        ":builder",
        ":load",
        ":save",
        ":signature_def_utils",
        ":simple_save",
        ":tag_constants",
        ":utils",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:backprop",
        "//machina/python/eager:lift_to_graph",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:function",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:versions",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:ref_variable",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/training:saver",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "load_optimizer_test",
    srcs = ["load_optimizer_test.py"],
    data = ["//machina/cc/saved_model:saved_model_test_files"],
    tags = ["no_oss"],  # Due to the usage of keras component.
    deps = [
        ":load",
        "//machina/python/eager:test",
        "//machina/python/ops:variables",
    ],
)

py_strict_library(
    name = "revived_types",
    srcs = ["revived_types.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/trackable:data_structures",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "revived_types_test",
    srcs = ["revived_types_test.py"],
    deps = [
        ":revived_types",
        "//machina/core:protos_all_py",
        "//machina/python/platform:client_testlib",
        "//machina/python/trackable:autotrackable",
    ],
)

py_strict_library(
    name = "function_serialization",
    srcs = ["function_serialization.py"],
    deps = [
        ":nested_structure_coder",
        "//machina/core:protos_all_py",
        "//machina/core/function/polymorphism:function_type",
        "//machina/python/eager:function",
        "//machina/python/eager:wrap_function",
        "//machina/python/eager/polymorphic_function:function_type_utils",
        "//machina/python/framework:func_graph",
        "//machina/python/util:nest",
    ],
)

py_strict_library(
    name = "function_deserialization",
    srcs = ["function_deserialization.py"],
    deps = [
        ":nested_structure_coder",
        "//machina/core:protos_all_py",
        "//machina/core/function/polymorphism:function_type",
        "//machina/core/function/trace_type",
        "//machina/python/eager:def_function",
        "//machina/python/eager:function",
        "//machina/python/eager/polymorphic_function:function_type_utils",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:function_def_to_graph",
        "//machina/python/framework:op_def_registry",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:type_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:custom_gradient",
        "//machina/python/ops:default_gradient",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_inspect",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "nested_structure_coder",
    srcs = ["nested_structure_coder.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:type_spec_registry",
        "//machina/python/types:internal",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "nested_structure_coder_test",
    srcs = ["nested_structure_coder_test.py"],
    deps = [
        ":nested_structure_coder",
        "//machina/core:protos_all_py",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:extension_type",
        "//machina/python/framework:immutable_dict",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:type_spec",
        "//machina/python/framework:type_spec_registry",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "//machina/python/types:internal",
    ],
)

py_strict_library(
    name = "save_options",
    srcs = ["save_options.py"],
    deps = [
        "//machina/python/checkpoint/sharding:sharding_util",
        "//machina/python/util:compat",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "load_options",
    srcs = ["load_options.py"],
    deps = [
        ":save_options",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "method_name_updater",
    srcs = ["method_name_updater.py"],
    deps = [
        ":constants",
        ":loader",
        "//machina/python/lib/io:file_io",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "method_name_updater_test",
    srcs = ["method_name_updater_test.py"],
    deps = [
        ":constants",
        ":loader",
        ":method_name_updater",
        "//machina/core:protos_all_py",
        "//machina/python/eager:test",
        "//machina/python/lib/io:file_io",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [
        ":builder",
        ":fingerprinting",
        ":load",
        ":load_v1_in_v2",
        ":loader",
        ":pywrap_saved_model",
        ":save",
        "//machina/python/checkpoint/sharding:sharding_policies",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/trackable:autotrackable",
    ],
)

tf_pybind_cc_library_wrapper(
    name = "pywrap_saved_model_headers",
    visibility = ["//visibility:private"],
    deps = [
        "//machina/cc/saved_model:constants",
        "//machina/cc/saved_model:fingerprinting",
        "//machina/cc/saved_model:metrics",
    ] + if_google([
        "//machina/tools/proto_splitter:merge",
    ]),
)

tf_python_pybind_extension(
    name = "pywrap_saved_model",
    srcs = [
        "pywrap_saved_model.cc",
        "pywrap_saved_model_constants.cc",
        "pywrap_saved_model_fingerprinting.cc",
        "pywrap_saved_model_metrics.cc",
    ] + if_google([
        "pywrap_saved_model_merger.cc",
    ]),
    hdrs = [
        "pywrap_saved_model_constants.h",
        "pywrap_saved_model_fingerprinting.h",
        "pywrap_saved_model_metrics.h",
    ] + if_google([
        "pywrap_saved_model_merger.h",
    ]),
    enable_stub_generation = True,
    # This fails Windows builds. Please check b/266870200 for details.
    #    dynamic_deps = ["//machina/python:_pywrap_machina_internal.so"] + select({
    #        "//machina:macos": ["//machina:libmachina_framework.%s.dylib" % VERSION],
    #        "//conditions:default": ["//machina:libmachina_framework.so.%s" % VERSION],
    #        "//machina:windows": [],
    #    }),
    #    static_deps = tf_python_pybind_static_deps(),
    pytype_srcs = [
        "pywrap_saved_model/__init__.pyi",
        "pywrap_saved_model/constants.pyi",
        "pywrap_saved_model/fingerprinting.pyi",
        "pywrap_saved_model/merger.pyi",
        "pywrap_saved_model/metrics.pyi",
    ],
    visibility = [
        "//machina/python:__pkg__",
        "//machina/python/checkpoint:__subpackages__",
        "//machina/python/tpu:__pkg__",
        "//machina/python/training:__subpackages__",
        "//machina/tools/pip_package:__subpackages__",
    ],
    deps = [
        # placeholder for index annotation deps
        "//machina/cc/experimental/libexport:save",
        "//machina/cc/saved_model:constants",
        "//machina/cc/saved_model:fingerprinting",
        "//machina/cc/saved_model:metrics",
        "//machina/cc/saved_model:reader",
        "//machina/core:core_cpu_base",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:path",
        "//machina/python/lib/core:pybind11_status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ] + if_google([
        "//machina/tools/proto_splitter:merge",
    ]),
)

tf_py_strict_test(
    name = "pywrap_saved_model_metrics_test",
    srcs = ["pywrap_saved_model_metrics_test.py"],
    deps = [
        ":pywrap_saved_model",
        "//machina/core:protos_all_py",
        "//machina/python/eager:test",
    ],
)

# copybara:uncomment_begin(google-only)
#
# tf_py_strict_test(
#     name = "pywrap_saved_model_merger_test",
#     srcs = ["pywrap_saved_model_merger_test.py"],
#     data = [
#         "//machina/tools/proto_splitter/testdata:split-standard.cpb",
#         "//machina/tools/proto_splitter/testdata:split-standard.pbtxt",
#     ],
#     deps = [
#         ":pywrap_saved_model",
#         "//machina/core:protos_all_py",
#         "//machina/python/eager:test",
#         "//machina/python/lib/io:file_io",
#         "//machina/python/platform:client_testlib",
#     ],
# )
#
# copybara:uncomment_end

tf_py_strict_test(
    name = "keras_injection_test",
    size = "small",
    srcs = ["keras_injection_test.py"],
    data = ["//machina/cc/saved_model:saved_model_test_files"],
    tags = ["no_oss"],  # Due to the usage of keras component.
    deps = [
        "//machina:machina_py",
        "//machina/python/eager:test",
    ],
)

tf_py_strict_test(
    name = "pywrap_saved_model_fingerprinting_test",
    srcs = ["pywrap_saved_model_fingerprinting_test.py"],
    data = ["//machina/cc/saved_model:saved_model_test_files"],
    tags = ["no_windows"],
    deps = [
        ":pywrap_saved_model",
        "//machina/core:protos_all_py",
        "//machina/python/eager:test",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "fingerprinting",
    srcs = ["fingerprinting.py"],
    deps = [
        ":pywrap_saved_model",
        "//machina/core:protos_all_py",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "fingerprinting_utils",
    srcs = ["fingerprinting_utils.py"],
    deps = [
        ":fingerprinting",
        ":pywrap_saved_model",
        "//machina/core:protos_all_py",
        "//machina/core/config:flags_py",
        "//machina/python/lib/io:file_io",
        "//machina/python/util:compat",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "fingerprinting_test",
    size = "small",
    srcs = ["fingerprinting_test.py"],
    deps = [
        ":fingerprinting",
        ":fingerprinting_utils",
        ":pywrap_saved_model",
        ":save",
        "//machina/core:protos_all_py",
        "//machina/core/config:flags_py",
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_spec",
        "//machina/python/lib/io:file_io",
        "//machina/python/trackable:autotrackable",
    ],
)

tf_py_strict_test(
    name = "tracing_utils_test",
    size = "small",
    srcs = ["tracing_utils_test.py"],
    deps = [
        ":tracing_utils",
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:variables",
        "//machina/python/trackable:base",
    ],
)

# copybara:uncomment_begin(google-only)
#
# tf_cc_test(
#     name = "lineage_logging_test",
#     srcs = ["lineage_logging_test.cc"],
#     deps = [
#         "//learning/brain/contrib/hub/public:client",
#         "//learning/brain/contrib/hub/public:handle",
#         "//learning/brain/contrib/hub/public/proto:metadata_cc_proto",
#         "//learning/brain/contrib/hub/public/proto:options_cc_proto",
#         "//learning/brain/contrib/hub/public/testing:in_process_tfhub_env",
#         "//learning/brain/contrib/hub/public/testing:testdata_util",
#         "//learning/metadata/cataline/proto:ml_data_processes_cc_proto",
#         "//learning/metadata/lineage_log/cc:lineage_log",
#         "//learning/metadata/lineage_log/cc:mock_process_client",
#         "//learning/metadata/lineage_log/cc:test_util",
#         "//machina/core/platform:protobuf",
#         "//machina/core/protobuf:for_core_protos_cc",
#         "@local_tsl//tsl/platform:env",
#         "@local_tsl//tsl/platform:path",
#         "//third_party/protobuf",
#         "//third_party/protobuf/io",
#         "@com_google_absl//absl/log:check",
#         "@com_google_absl//absl/memory",
#         "@com_google_absl//absl/status",
#         "@com_google_absl//absl/strings",
#         "@com_google_absl//absl/strings:string_view",
#         "@com_google_googletest//:gtest_main",
#     ],
# )
#
# copybara:uncomment_end
