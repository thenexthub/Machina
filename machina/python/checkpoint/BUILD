# Description:
#   Utilities for reading and writing object-based checkpoints.

load("//machina:strict.default.bzl", "py_strict_binary", "py_strict_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")
load(
    "//machina/tools/test:performance.bzl",
    "tf_py_benchmark_test",
    "tf_py_logged_benchmark",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "checkpoint_lib",
    visibility = [
        "//machina:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":checkpoint",
        ":checkpoint_management",
        ":checkpoint_options",
        ":functional_saver",
        ":graph_view",
        ":saveable_compat",
        ":util",
    ],
)

py_strict_library(
    name = "checkpoint_adapter",
    srcs = ["checkpoint_adapter.py"],
    deps = [
        "//machina/python/framework:tensor",
        "//machina/python/trackable:base",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "async_checkpoint_helper",
    srcs = ["async_checkpoint_helper.py"],
    deps = [
        ":checkpoint_context",
        ":trackable_view",
        "//machina/python/distribute:device_util",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:executor",
        "//machina/python/framework:ops",
        "//machina/python/ops:variables",
        "//machina/python/saved_model:pywrap_saved_model",
        "//machina/python/trackable:base",
        "//machina/python/util:object_identity",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "checkpoint",
    srcs = ["checkpoint.py"],
    deps = [
        ":async_checkpoint_helper",
        ":checkpoint_context",
        ":checkpoint_management",
        ":checkpoint_options",
        ":functional_saver",
        ":graph_view",
        ":restore",
        ":save_util",
        ":save_util_v1",
        ":util",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:monitoring",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:io_ops_gen",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:path_helpers",
        "//machina/python/saved_model:pywrap_saved_model",
        "//machina/python/trackable:autotrackable",
        "//machina/python/trackable:base",
        "//machina/python/trackable:data_structures",
        "//machina/python/training:py_checkpoint_reader",
        "//machina/python/training:saver",
        "//machina/python/training/saving:saveable_object",
        "//machina/python/training/saving:saveable_object_util",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_contextlib",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
    ],
)

tf_py_strict_test(
    name = "checkpoint_test",
    srcs = ["checkpoint_test.py"],
    tags = [
        "no_pip",  # TODO(b/250108043)
        "no_windows",  # TODO(b/201457117)
    ],
    deps = [
        ":async_checkpoint_helper",
        ":checkpoint",
        ":checkpoint_management",
        ":checkpoint_options",
        ":graph_view",
        ":save_util",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:stack",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:template",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:save",
        "//machina/python/trackable:autotrackable",
        "//machina/python/trackable:base",
        "//machina/python/training:adam",
        "//machina/python/training:checkpoint_utils",
        "//machina/python/training:saver",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "checkpoint_context",
    srcs = [
        "checkpoint_context.py",
    ],
)

tf_py_strict_test(
    name = "checkpoint_with_v1_optimizers_test",
    srcs = ["checkpoint_with_v1_optimizers_test.py"],
    deps = [
        ":checkpoint",
        ":checkpoint_options",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/eager:test",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:template",
        "//machina/python/ops:variable_scope",
        "//machina/python/trackable:autotrackable",
        "//machina/python/training:adam",
        "//machina/python/training:checkpoint_utils",
    ],
)

tf_py_strict_test(
    name = "checkpoint_metrics_test",
    srcs = ["checkpoint_metrics_test.py"],
    deps = [
        ":checkpoint",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:pywrap_saved_model",
    ],
)

py_strict_library(
    name = "checkpoint_view",
    srcs = ["checkpoint_view.py"],
    tags = ["no_pip"],
    deps = [
        ":trackable_view",
        "//machina/core:protos_all_py",
        "//machina/python/framework:errors",
        "//machina/python/platform:tf_logging",
        "//machina/python/trackable:base",
        "//machina/python/training:py_checkpoint_reader",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "checkpoint_view_test",
    srcs = ["checkpoint_view_test.py"],
    tags = ["no_pip"],
    deps = [
        ":checkpoint",
        ":checkpoint_view",
        "//machina/python/eager:test",
        "//machina/python/trackable:autotrackable",
    ],
)

py_strict_library(
    name = "graph_view",
    srcs = ["graph_view.py"],
    deps = [
        ":save_util_v1",
        ":trackable_view",
        "//machina/python/trackable:base",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "save_util",
    srcs = ["save_util.py"],
    deps = [
        ":graph_view",
        ":save_util_v1",
        ":saveable_compat",
        ":util",
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/saved_model/registration",
        "//machina/python/trackable:base",
        "//machina/python/trackable:python_state",
        "//machina/python/trackable:trackable_utils",
        "//machina/python/training/saving:saveable_object",
        "//machina/python/training/saving:saveable_object_util",
        "//machina/python/types:core",
        "//machina/python/util:object_identity",
    ],
)

py_strict_library(
    name = "save_util_v1",
    srcs = ["save_util_v1.py"],
    deps = [
        ":saveable_compat",
        ":util",
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/saved_model/registration",
        "//machina/python/trackable:base",
        "//machina/python/trackable:python_state",
        "//machina/python/trackable:trackable_utils",
        "//machina/python/training/saving:saveable_object",
        "//machina/python/training/saving:saveable_object_util",
        "//machina/python/util:object_identity",
    ],
)

tf_py_strict_test(
    name = "save_util_v1_test",
    srcs = ["save_util_v1_test.py"],
    deps = [
        ":graph_view",
        ":save_util_v1",
        "//machina/python/eager:test",
        "//machina/python/ops:variables",
        "//machina/python/saved_model/registration",
        "//machina/python/trackable:autotrackable",
        "//machina/python/util:object_identity",
    ],
)

py_strict_library(
    name = "trackable_view",
    srcs = ["trackable_view.py"],
    tags = ["no_pip"],
    deps = [
        "//machina/python/trackable:base",
        "//machina/python/trackable:converter",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "trackable_view_test",
    srcs = ["trackable_view_test.py"],
    deps = [
        ":trackable_view",
        "//machina/python/eager:test",
        "//machina/python/trackable:base",
    ],
)

py_strict_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/trackable:trackable_utils",
        "//machina/python/util:object_identity",
    ],
)

py_strict_library(
    name = "restore",
    srcs = ["restore.py"],
    deps = [
        ":checkpoint_adapter",
        ":checkpoint_view",
        ":functional_saver",
        ":save_util_v1",
        ":saveable_compat",
        "//machina/python/eager:context",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:io_ops_gen",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model/registration",
        "//machina/python/trackable:base",
        "//machina/python/trackable:constants",
        "//machina/python/trackable:python_state",
        "//machina/python/trackable:trackable_utils",
        "//machina/python/training/saving:saveable_object_util",
        "//machina/python/util:object_identity",
    ],
)

tf_py_strict_test(
    name = "restore_test",
    srcs = ["restore_test.py"],
    deps = [
        ":checkpoint",
        ":restore",
        "//machina/python/eager:test",
        "//machina/python/module",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:variables",
        "//machina/python/trackable:autotrackable",
        "//machina/python/trackable:base",
        "//machina/python/training/saving:saveable_object",
    ],
)

tf_py_benchmark_test(
    name = "benchmarks_test",
    srcs = ["benchmarks_test.py"],
    deps = [
        ":checkpoint",
        "//machina/python/framework:ops",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:io_ops_gen",
        "//machina/python/platform:client_testlib",
        "//machina/python/trackable:base",
        "//machina/python/training:py_checkpoint_reader",
    ],
)

tf_py_logged_benchmark(
    name = "benchmarks",
    target = "//machina/python/checkpoint:benchmarks_test",
)

py_strict_library(
    name = "checkpoint_options",
    srcs = ["checkpoint_options.py"],
    deps = [
        "//machina/python/checkpoint/sharding:sharding_util",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "functional_saver",
    srcs = ["functional_saver.py"],
    deps = [
        ":checkpoint_options",
        "//machina/core:protos_all_py",
        "//machina/python/checkpoint/sharding:sharding_policies",
        "//machina/python/checkpoint/sharding:sharding_util",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:device",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:io_ops_gen",
        "//machina/python/ops:string_ops",
        "//machina/python/saved_model:pywrap_saved_model",
        "//machina/python/saved_model/registration",
        "//machina/python/trackable:base",
        "//machina/python/trackable:trackable_utils",
        "//machina/python/training/saving:saveable_object",
        "//machina/python/training/saving:saveable_object_util",
        "//machina/python/types:core",
        "//machina/python/util:nest",
        "//machina/python/util:object_identity",
        "@absl_py//absl/logging",
    ],
)

cuda_py_strict_test(
    name = "functional_saver_test",
    size = "medium",
    srcs = [
        "functional_saver_test.py",
    ],
    deps = [
        ":checkpoint",
        ":checkpoint_options",
        ":functional_saver",
        ":graph_view",
        "//machina/python/eager:context",
        "//machina/python/eager:remote",
        "//machina/python/eager:test",
        "//machina/python/eager:wrap_function",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/module",
        "//machina/python/ops:io_ops_gen",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/platform:gfile",
        "//machina/python/training:server_lib",
        "//machina/python/training/saving:saveable_object_util",
    ],
)

py_strict_library(
    name = "tensor_callable",
    srcs = ["tensor_callable.py"],
    deps = [
        "//machina/python/training/saving:saveable_object",
    ],
)

tf_py_strict_test(
    name = "tensor_callable_test",
    srcs = ["tensor_callable_test.py"],
    deps = [
        ":checkpoint",
        ":tensor_callable",
        "//machina/python/eager:test",
        "//machina/python/ops:variables",
        "//machina/python/saved_model:save",
        "//machina/python/trackable:base",
    ],
)

py_strict_library(
    name = "checkpoint_management",
    srcs = ["checkpoint_management.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/checkpoint:checkpoint_options",
        "//machina/python/eager:context",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:variable_scope",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:checkpoint_state_py",
        "//machina/python/training:training_util",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "checkpoint_management_test",
    size = "small",
    srcs = [
        "checkpoint_management_test.py",
    ],
    deps = [
        ":checkpoint",
        ":checkpoint_management",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:checkpoint_state_py",
        "//machina/python/training:saver",
    ],
)

py_strict_library(
    name = "saveable_compat",
    srcs = [
        "saveable_compat.py",
    ],
)

tf_py_strict_test(
    name = "saveable_compat_test",
    srcs = [
        "saveable_compat_test.py",
    ],
    data = [
        "testdata/table_legacy_saveable_object.data-00000-of-00001",
        "testdata/table_legacy_saveable_object.index",
    ],
    tags = ["no_pip"],
    deps = [
        ":checkpoint",
        ":generate_checkpoint_lib",
        ":saveable_compat",
        "//machina/python/eager:test",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:variables",
        "//machina/python/trackable:base",
        "//machina/python/training:checkpoint_utils",
        "//machina/python/training/saving:saveable_object",
    ],
)

py_strict_binary(
    name = "generate_checkpoint",
    srcs = ["testdata/generate_checkpoint.py"],
    deps = [
        "//machina/python/checkpoint",
        "//machina/python/compat:v2_compat",
        "//machina/python/framework:dtypes",
        "//machina/python/module",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:variables",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "generate_checkpoint_lib",
    srcs = ["testdata/generate_checkpoint.py"],
    deps = [
        "//machina/python/checkpoint",
        "//machina/python/compat:v2_compat",
        "//machina/python/framework:dtypes",
        "//machina/python/module",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:variables",
        "@absl_py//absl:app",
    ],
)
