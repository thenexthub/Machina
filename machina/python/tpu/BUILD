load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

# Description: Operations defined for Cloud TPUs
load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

# Placeholder: load py_proto_library
load("//machina:machina.default.bzl", "tf_py_strict_test", "tf_python_pybind_extension")
load("//machina/core/platform:build_config.bzl", "tf_proto_library")
load("//machina/core/platform:build_config_root.bzl", "if_pywrap")
load("//machina/python/tpu:tpu.bzl", "internal_create_sanitizer_settings", "tpu_py_strict_test")

# Do not add anymore paths here. You do not need to be in the visibility list
# to use TPU symbols. They are accessible from tf.contrib.tpu in TF 1.x and
# tf.tpu and tf.compat.v1.tpu in TF 2.x.
visibility = [
    "//learning/brain:__subpackages__",
    "//learning/deepmind:__subpackages__",
    "//learning/serving:__subpackages__",
    "//research/graph:__subpackages__",
    "//smartass/brain/configure:__subpackages__",
    "//third_party/py/jax_tpu_embedding:__subpackages__",
    "//third_party/py/lingvo:__subpackages__",
    "//third_party/py/orbit:__subpackages__",
    "//third_party/py/medical_research_foundations:__subpackages__",
    "//machina:__subpackages__",
    "//waymo/ml/deploy/sync_test/tools:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = visibility + ["//machina/tools/pip_package:__subpackages__"],
    licenses = ["notice"],
)

exports_files(["tpu_test_wrapper.py"])

bzl_library(
    name = "tpu_test_wrapper_bzl",
    srcs = ["tpu_test_wrapper.bzl"],
)

py_strict_library(
    name = "tpu_test_wrapper",
    srcs = ["tpu_test_wrapper.py"],
    deps = [
        "//machina/python/platform:flags",
        "//machina/python/util:tf_inspect",
    ],
)

py_strict_library(
    name = "embedding_context_utils",
    srcs = ["embedding_context_utils.py"],
    deps = [
        "@absl_py//absl/logging",
    ],
)

py_strict_test(
    name = "tpu_test_wrapper_test",
    srcs = ["tpu_test_wrapper_test.py"],
    main = "tpu_test_wrapper_test.py",
    tags = [
        "no_oss_py35",
        "no_pip",
    ],
    deps = [
        ":tpu_test_wrapper",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:flags",
        "@absl_py//absl/testing:flagsaver",
    ],
)

tf_py_strict_test(
    name = "tpu_embedding_v3_cpu_ops_test",
    srcs = ["tpu_embedding_v3_cpu_ops_test.py"],
    shard_count = 8,
    tags = ["no_oss"],
    deps = [
        # copybara:uncomment "//machina/core/tpu/kernels",
        "//machina/python/compat:v2_compat",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:bitwise_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/tpu/ops:gen_xla_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

alias(
    name = "tpu_ops",
    actual = "//machina/python/tpu/ops",
)

pytype_strict_library(
    name = "async_checkpoint",
    srcs = ["async_checkpoint.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:pywrap_saved_model",
        "//machina/python/training",
        "//machina/python/training:basic_session_run_hooks",
        "//machina/python/training:monitored_session",
        "//machina/python/training:saver",
        "//machina/python/training:session_run_hook",
        "//machina/python/training:summary_io",
        "//machina/python/training:training_util",
    ],
)

tpu_py_strict_test(
    name = "async_checkpoint_test",
    size = "medium",
    srcs = ["async_checkpoint_test.py"],
    disable_experimental = True,
    disable_mlir_bridge = False,
    tags = ["no_oss"],
    deps = [
        ":async_checkpoint",
        ":tpu_lib",
        "//machina/core:protos_all_py",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:metrics",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops/losses",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:flags",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:pywrap_saved_model",
        "//machina/python/training:basic_session_run_hooks",
        "//machina/python/training:training_lib",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "device_assignment",
    srcs = ["device_assignment.py"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":topology",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:numpy_compat",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "preempted_hook_py",
    srcs = ["preempted_hook.py"],
    deps = [
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:session_run_hook",
    ],
)

py_strict_library(
    name = "tpu_replication",
    srcs = ["tpu_replication.py"],
    deps = [
        ":device_assignment",
        "//machina/core:protos_all_py",
        "//machina/python/distribute:device_util",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/framework:device",
        "//machina/python/framework:errors",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:variables",
        "//machina/python/tpu/ops",
        "//machina/python/types:core",
        "//machina/python/util:compat",
        "//machina/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "functional",
    srcs = ["functional.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//machina/python/tpu/ops",
    ],
)

pytype_strict_library(
    name = "topology",
    srcs = ["topology.py"],
    deps = [
        "//machina/core/protobuf/tpu:topology_proto_py",
        "//machina/python/util:numpy_compat",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "tpu",
    srcs = [
        "__init__.py",
    ],
    deps = ["//machina/python/framework:versions"],
)

py_strict_library(
    name = "tpu_noestimator",
    srcs = [
        "__init__.py",
        "api.py",
    ],
    deps = [
        ":feature_column_v2",
        ":tpu_embedding_for_serving",
        ":tpu_embedding_v1",
        ":tpu_embedding_v2",
        ":tpu_embedding_v2_utils",
        ":tpu_embedding_v3",
        ":tpu_hardware_feature",
        ":tpu_lib",
        ":tpu_py",
        "//machina/python/framework:versions",
    ],
)

pytype_strict_library(
    name = "tensor_tracer",
    srcs = ["tensor_tracer.py"],
    deps = [
        ":tensor_tracer_flags",
        ":tensor_tracer_report",
        ":tpu_replication",
        "//machina/core:protos_all_py",
        "//machina/python/eager:monitoring",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:function",
        "//machina/python/framework:graph_io",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_util",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_case",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:control_flow_util",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:logging_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:nn_impl",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:summary_ops_v2",
        "//machina/python/ops:variable_scope",
        "//machina/python/platform:analytics",
        "//machina/python/platform:gfile",
        "//machina/python/platform:remote_utils",
        "//machina/python/platform:tf_logging",
        "//machina/python/summary:summary_iterator",
        "//machina/python/tpu/ops",
        "//machina/python/training:training_util",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "tensor_tracer_report",
    srcs = ["tensor_tracer_report.py"],
    deps = [
        ":tensor_tracer_proto_py",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
    ],
)

pytype_strict_library(
    name = "tensor_tracer_flags",
    srcs = ["tensor_tracer_flags.py"],
    deps = [
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:tf_logging",
        "@absl_py//absl/flags",
    ],
)

pytype_strict_library(
    name = "tpu_lib",
    srcs = [
        "__init__.py",
        "bfloat16.py",
        "tpu_optimizer.py",
        "tpu_strategy_util.py",
        "training_loop.py",
    ],
    deps = [
        ":tensor_tracer",
        ":topology",
        ":tpu_feed",
        ":tpu_function",
        ":tpu_py",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/compiler/xla",
        "//machina/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:monitoring",
        "//machina/python/framework:device",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:versions",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:while_loop",
        "//machina/python/ops/losses",
        "//machina/python/platform:tf_logging",
        "//machina/python/tpu/ops",
        "//machina/python/training:optimizer",
        "//machina/python/training:session_run_hook",
        "//machina/python/training:training_util",
        "//machina/python/types:core",
        "//machina/python/util:compat",
        "//machina/python/util:tf_contextlib",
        "//machina/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "tpu_py",
    srcs = ["tpu.py"],
    deps = [
        ":device_assignment",
        ":tensor_tracer",
        ":tpu_feed",
        ":tpu_function",
        ":tpu_name_util",
        ":tpu_replication",
        "//machina/compiler/tf2xla/python:xla",
        "//machina/core:protos_all_py",
        "//machina/core/protobuf/tpu:compilation_result_proto_py",
        "//machina/core/protobuf/tpu:dynamic_padding_proto_py",
        "//machina/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
        "//machina/python:tf2",
        "//machina/python/compiler/xla",
        "//machina/python/framework:auto_control_deps",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:function",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/tpu/ops",
        "//machina/python/types:core",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_export",
        "//machina/python/util:traceback_utils",
        "//machina/python/util:variable_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "tpu_feed",
    srcs = ["tpu_feed.py"],
    deps = [
        ":tpu_name_util",
        ":tpu_sharding",
        "//machina/python/compiler/xla/experimental:xla_sharding",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/tpu/ops",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "tpu_function",
    srcs = ["tpu_function.py"],
)

pytype_strict_library(
    name = "tpu_sharding",
    srcs = ["tpu_sharding.py"],
    deps = [
        "//machina/python/framework:tensor_shape",
    ],
)

pytype_strict_library(
    name = "tpu_system_metadata",
    srcs = ["tpu_system_metadata.py"],
    deps = [
        ":tpu_py",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/distribute:device_util",
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:device",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "datasets",
    srcs = [
        "datasets.py",
    ],
    visibility = visibility,
    deps = [
        "//machina/python/data/experimental/ops:interleave_ops",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:iterator_ops",
        "//machina/python/data/ops:readers",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:function",
        "//machina/python/framework:ops",
        "//machina/python/ops:functional_ops",
        "//machina/python/types:data",
    ],
)

tf_py_strict_test(
    name = "datasets_test",
    size = "medium",
    srcs = ["datasets_test.py"],
    grpc_enabled = True,
    shard_count = 4,
    tags = ["no_oss"],
    deps = [
        ":datasets",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:readers",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/lib/io:python_io",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:server_lib",
        "//machina/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "tpu_test",
    size = "small",
    srcs = ["tpu_test.py"],
    tags = [
        "no_oss",  # TODO(b/131157871): Reenable in OSS when fixed
        "no_windows",  # TODO: needs investigation on Windows
    ],
    deps = [
        ":tpu_feed",
        ":tpu_lib",
        ":tpu_py",
        ":tpu_replication",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/layers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:control_flow_util",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:special_math_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/platform:client_testlib",
        "//machina/python/tpu/ops",
    ],
)

tf_py_strict_test(
    name = "tpu_sharding_test",
    size = "small",
    srcs = ["tpu_sharding_test.py"],
    deps = [
        ":tpu_sharding",
        "//machina/python/framework:tensor_shape",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "bfloat16_test",
    size = "small",
    srcs = ["bfloat16_test.py"],
    deps = [
        ":tpu_lib",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "tpu_infeed_test",
    size = "small",
    srcs = ["tpu_infeed_test.py"],
    deps = [
        ":tpu_feed",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "topology_test",
    size = "medium",
    srcs = ["topology_test.py"],
    deps = [
        ":topology",
        "//machina/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "tpu_strategy_util",
    srcs = ["tpu_strategy_util.py"],
    visibility = [
        "//machina:__subpackages__",
        "//third_party/py/machina_numerics/extensions:__pkg__",
    ],
    deps = [
        ":topology",
        ":tpu_py",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:monitoring",
        "//machina/python/framework:device",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
    ],
)

pytype_strict_library(
    name = "tpu_hardware_feature",
    srcs = ["tpu_hardware_feature.py"],
    deps = [
        "//machina/core/protobuf/tpu:topology_proto_py",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "tpu_name_util",
    srcs = ["tpu_name_util.py"],
    deps = [
        "//machina/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "feature_column",
    srcs = ["feature_column.py"],
    deps = [
        ":tpu_function",
        ":tpu_py",
        ":tpu_replication",
        "//machina/python/feature_column",
        "//machina/python/feature_column:feature_column_py",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:variable_scope",
    ],
)

pytype_strict_library(
    name = "feature_column_v2",
    srcs = ["feature_column_v2.py"],
    deps = [
        ":feature_column",
        ":tpu_py",
        ":tpu_replication",
        "//machina/python/feature_column",
        "//machina/python/feature_column:feature_column_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:embedding_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "feature_column_test",
    srcs = [
        "feature_column_test.py",
    ],
    main = "feature_column_test.py",
    deps = [
        ":feature_column",
        "//machina/python/client:session",
        "//machina/python/feature_column",
        "//machina/python/feature_column:feature_column_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "feature_column_v2_test",
    srcs = [
        "feature_column_v2_test.py",
    ],
    main = "feature_column_v2_test.py",
    tags = ["no_oss"],  # Due to the usage of keras component.
    deps = [
        ":feature_column_v2",
        ":tpu_function",
        ":tpu_py",
        "//machina/python/client:session",
        "//machina/python/feature_column:feature_column_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_v2_utils",
    srcs = ["tpu_embedding_v2_utils.py"],
    deps = [
        "//machina/core/protobuf/tpu:optimization_parameters_proto_py",
        "//machina/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
        "//machina/python/distribute:device_util",
        "//machina/python/distribute:sharded_variable",
        "//machina/python/distribute:tpu_strategy",
        "//machina/python/framework:device_spec",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/tpu/ops",
        "//machina/python/types:core",
        "//machina/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_v2",
    srcs = ["tpu_embedding_v2.py"],
    deps = [
        ":tpu_embedding_v2_utils",
        ":tpu_py",
        ":tpu_replication",
        "//machina/core:protos_all_py",
        "//machina/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
        "//machina/python/distribute:device_util",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:distribute_utils",
        "//machina/python/distribute:sharded_variable",
        "//machina/python/distribute:tpu_strategy",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:device",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/saved_model:save_context",
        "//machina/python/saved_model/registration",
        "//machina/python/tpu/ops",
        "//machina/python/trackable:autotrackable",
        "//machina/python/trackable:base",
        "//machina/python/types:internal",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_base",
    srcs = ["tpu_embedding_base.py"],
    deps = [
        ":tpu_embedding_v2_utils",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:variables",
        "//machina/python/trackable:autotrackable",
        "//machina/python/util:nest",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_v3_utils",
    srcs = ["tpu_embedding_v3_utils.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops_gen",
        "//machina/python/ops:manip_ops",
        "//machina/python/ops:variables",
        "//machina/python/trackable:base",
    ],
)

tf_py_strict_test(
    name = "tpu_embedding_v3_utils_test",
    srcs = ["tpu_embedding_v3_utils_test.py"],
    deps = [
        ":tpu_embedding_v3_utils",
        "//machina/python/compat:v2_compat",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_v3_checkpoint_adapter",
    srcs = ["tpu_embedding_v3_checkpoint_adapter.py"],
    deps = [
        ":tpu_embedding_v3_utils",
        "//machina/core/tpu/kernels:sparse_core_layout_proto_py",
        "//machina/python/checkpoint:checkpoint_adapter",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/trackable:base",
        "//machina/python/training:py_checkpoint_reader",
        "//machina/python/util/protobuf",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "tpu_embedding_v3_checkpoint_adapter_test",
    srcs = ["tpu_embedding_v3_checkpoint_adapter_test.py"],
    deps = [
        ":tpu_embedding_v3_checkpoint_adapter",
        "//machina/core/tpu/kernels:sparse_core_layout_proto_py",
        "//machina/python/compat:v2_compat",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_for_serving",
    srcs = ["tpu_embedding_for_serving.py"],
    deps = [
        ":tpu_embedding_base",
        ":tpu_embedding_v2_utils",
        ":tpu_embedding_v3_utils",
        "//machina/core/tpu/kernels:sparse_core_layout_proto_py",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:tpu_strategy",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:embedding_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/trackable:base",
        "//machina/python/trackable:constants",
        "//machina/python/types:core",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "tpu_embedding_for_serving_test",
    srcs = [
        "tpu_embedding_for_serving_test.py",
    ],
    deps = [
        ":tpu_embedding_for_serving",
        ":tpu_embedding_v2_utils",
        "//machina/python/compat:v2_compat",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops:init_ops_v2",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_v1",
    srcs = ["tpu_embedding_v1.py"],
    deps = [
        ":tpu_embedding_base",
        ":tpu_embedding_v2_utils",
        ":tpu_replication",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:tpu_strategy",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:embedding_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "tpu_embedding_v2_utils_test",
    srcs = [
        "tpu_embedding_v2_utils_test.py",
    ],
    deps = [
        ":tpu_embedding_v2_utils",
        "//machina/core/protobuf/tpu:tpu_embedding_configuration_proto_py",
        "//machina/python/compat:v2_compat",
        "//machina/python/eager:def_function",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_outside_compilation_test",
    srcs = [
        "tpu_outside_compilation_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    deps = [
        ":functional",
        ":tpu_py",
        ":tpu_replication",
        "//machina/core:protos_all_py",
        "//machina/python/distribute:tpu_strategy",
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/eager:def_function",
        "//machina/python/eager:remote",
        "//machina/python/eager:test",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/lib/io:tf_record",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:logging_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:summary_ops_v2",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:flags",
        "//machina/python/platform:gfile",
        "//machina/python/tpu/ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

# NOTE this target should only be depended on by the tpu_test_wrapper macro.
py_strict_library(
    name = "tpu_test_deps",
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:flags",
        "//machina/python/util:tf_inspect",
    ],
)

tf_proto_library(
    name = "tensor_tracer_proto",
    srcs = ["tensor_tracer.proto"],
    protodeps = [
        "//machina/core:protos_all",
    ],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "tensor_tracer_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":tensor_tracer_proto"],
# )
# copybara:uncomment_end

tf_python_pybind_extension(
    name = "_pywrap_sparse_core_layout",
    srcs = ["pywrap_sparse_core_layout.cc"],
    enable_stub_generation = True,
    features = ["-layering_check"],
    pytype_srcs = [
        "_pywrap_sparse_core_layout.pyi",
    ],
    deps = [
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_status",
        "//machina/python/lib/core:pybind11_status_headers",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ] + if_pywrap(
        if_false = [
            "//machina/core/tpu/kernels:_pywrap_sparse_core_layout_header_only",
        ],
        if_true = [
            "//machina/core/tpu/kernels:sparse_core_layout",
        ],
    ),
)

tf_python_pybind_extension(
    name = "_pywrap_tpu_embedding",
    srcs = ["pywrap_tpu_embedding.cc"],
    enable_stub_generation = True,
    features = ["-layering_check"],
    pytype_srcs = [
        "_pywrap_tpu_embedding.pyi",
    ],
    deps = [
        "//machina/core/tpu/kernels:sparse_core_ops_utils",
        "//machina/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ],
)

py_strict_library(
    name = "tpu_embedding_v3",
    srcs = ["tpu_embedding_v3.py"],
    deps = [
        ":_pywrap_sparse_core_layout",  # buildcleaner: keep
        ":_pywrap_tpu_embedding",  # buildcleaner: keep
        ":embedding_context_utils",
        ":tpu_embedding_base",
        ":tpu_embedding_v2_utils",
        ":tpu_embedding_v3_checkpoint_adapter",
        ":tpu_embedding_v3_utils",
        ":tpu_replication",
        "//machina/core:protos_all_py",
        "//machina/core/tpu/kernels:sparse_core_layout_proto_py",
        "//machina/python/checkpoint:saveable_compat",
        "//machina/python/distribute:device_util",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:tpu_strategy",
        "//machina/python/distribute:tpu_util",
        "//machina/python/distribute:tpu_values",
        "//machina/python/distribute:values",
        "//machina/python/distribute:values_util",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:device",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops_gen",
        "//machina/python/ops:summary_ops_v2",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/tpu/ops:gen_xla_ops",
        "//machina/python/trackable:base",
        "//machina/python/training/saving:saveable_object",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
        "@absl_py//absl/logging",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v3_test",
    srcs = ["tpu_embedding_v3_test.py"],
    disable_experimental = True,
    disable_tfrt = False,
    disable_v2 = True,
    disable_v3 = True,
    # copybara:uncomment disable_v4i = True,
    # copybara:uncomment disable_v5 = False,
    # copybara:uncomment disable_v5_grm = True,
    # copybara:uncomment disable_v6e = True,
    shard_count = 4,
    # copybara:uncomment use_tpu_plugin_and_capi = True,
    deps = [
        ":embedding_context_utils",
        ":tpu_embedding_for_serving",
        ":tpu_embedding_v2_utils",
        ":tpu_embedding_v3",
        ":tpu_embedding_v3_utils",
        ":tpu_replication",
        "//machina/python/compat:v2_compat",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:tpu_strategy",
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/eager:def_function",
        "//machina/python/eager:remote",
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:summary_ops_v2",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v3_additional_test",
    srcs = ["tpu_embedding_v3_additional_test.py"],
    args = [
        "--tpu=''",
    ],
    disable_experimental = True,
    disable_tfrt = False,
    disable_v2 = True,
    disable_v3 = True,
    # copybara:uncomment disable_v4i = True,
    # copybara:uncomment disable_v5 = False,
    # copybara:uncomment disable_v5_grm = True,
    # copybara:uncomment disable_v6e = False,
    env = {
        "TF_MACHINA_MACHINA_XLA_FLAGS": (
            "--tf_mlir_enable_mlir_bridge=true " +
            "--tf_mlir_enable_convert_control_to_data_outputs_pass=true " +
            "--tf_mlir_enable_merge_control_flow_pass=true " +
            "--tf_mlir_enable_strict_clusters=true " +
            # These tests run lookup without while loops.
            "--tf_xla_disable_full_embedding_pipelining=true"
        ),
    },
    # Make it so there are enough shards so each test runs in separate shards.
    shard_count = 12,
    visibility = ["//visibility:private"],
    deps = [
        ":device_assignment",
        ":tpu_embedding_for_serving",
        ":tpu_embedding_v2_utils",
        ":tpu_embedding_v3",
        ":tpu_embedding_v3_utils",
        "//machina/python/checkpoint",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:tpu_strategy",
        "//machina/python/distribute:values",
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/eager:def_function",
        "//machina/python/eager:remote",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:save",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v3_checkpoint_test",
    srcs = ["tpu_embedding_v3_checkpoint_test.py"],
    disable_experimental = True,
    disable_tfrt = False,
    disable_v2 = True,
    disable_v3 = True,
    # copybara:uncomment disable_v4i = True,
    # copybara:uncomment disable_v5 = False,
    # copybara:uncomment disable_v5_grm = True,
    # copybara:uncomment disable_v6e = False,
    deps = [
        ":tpu_embedding_v2_utils",
        ":tpu_embedding_v3",
        "//machina/python/checkpoint",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute:tpu_strategy",
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/framework:config",
        "//machina/python/ops:init_ops_v2",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

internal_create_sanitizer_settings()
