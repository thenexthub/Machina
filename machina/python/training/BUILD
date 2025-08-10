load("//machina:strict.default.bzl", "py_strict_library")

# Placeholder: load py_proto_library
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")
load("//machina/core/platform:build_config.bzl", "tf_proto_library")
load("//machina/core/platform:distribute.bzl", "distribute_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

exports_files(
    # Used in a pybind extension whose rule must be in machina/python
    ["quantize_training_wrapper.cc"],
    visibility = ["//machina/python:__pkg__"],
)

py_strict_library(
    name = "training_lib",
    srcs = ["training.py"],
    visibility = [
        "//machina:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":adadelta",
        ":adagrad",
        ":adagrad_da",
        ":adam",
        ":basic_loops",
        ":basic_session_run_hooks",
        ":checkpoint_utils",
        ":coordinator",
        ":device_setter",
        ":ftrl",
        ":gradient_descent",
        ":input",
        ":learning_rate_decay",
        ":momentum",
        ":monitored_session",
        ":moving_averages",
        ":optimizer",
        ":proximal_adagrad",
        ":proximal_gradient_descent",
        ":py_checkpoint_reader",
        ":queue_runner",
        ":rmsprop",
        ":saver",
        ":server_lib",
        ":session_manager",
        ":session_run_hook",
        ":summary_io",
        ":supervisor",
        ":sync_replicas_optimizer",
        ":training_util",
        ":warm_starting_util",
        "//machina/core:protos_all_py",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint:checkpoint_management",
        "//machina/python/checkpoint:checkpoint_view",
        "//machina/python/ops:sdca_ops",
        "//machina/python/trackable:python_state",
        "//machina/python/training/experimental:loss_scale_optimizer",
        "//machina/python/training/experimental:mixed_precision",
        "//machina/python/training/saving:saveable_object_util",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "training",
    visibility = [
        "//machina:internal",
        "//machina_minigo:__subpackages__",
        "//machina_models:__subpackages__",
        "//third_party/cloud_tpu/convergence_tools:__subpackages__",
        "//third_party/mlperf:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":training_lib",
        "//machina/python/checkpoint",
        "//machina/python/trackable:base",
        "//machina/python/trackable:python_state",
    ],
)

py_strict_library(
    name = "adadelta",
    srcs = ["adadelta.py"],
    deps = [
        ":optimizer",
        "//machina/python/framework:ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:training_ops_gen",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "adagrad_da",
    srcs = ["adagrad_da.py"],
    deps = [
        ":optimizer",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:training_ops_gen",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "adagrad",
    srcs = ["adagrad.py"],
    deps = [
        ":optimizer",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:training_ops_gen",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "adam",
    srcs = ["adam.py"],
    deps = [
        ":optimizer",
        "//machina/python/eager:context",
        "//machina/python/framework:ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:training_ops_gen",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "basic_loops",
    srcs = ["basic_loops.py"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "checkpoint_ops",
    srcs = ["checkpoint_ops.py"],
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:checkpoint_ops_gen",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
    ],
)

py_strict_library(
    name = "checkpoint_utils",
    srcs = ["checkpoint_utils.py"],
    deps = [
        ":py_checkpoint_reader",
        "//machina/python/checkpoint:checkpoint_management",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/framework:ops",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/training/saving:saveable_object_util",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "coordinator",
    srcs = ["coordinator.py"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "device_setter",
    srcs = ["device_setter.py"],
    visibility = [
        "//machina:internal",
        "//third_party/py/tf_slim/ops:__pkg__",
    ],
    deps = [
        ":server_lib",
        "//machina/core:protos_all_py",
        "//machina/python/framework:device",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    visibility = [
        "//machina:internal",
        "//third_party/py/tf_slim/training:__pkg__",
    ],
    deps = [
        ":basic_session_run_hooks",
        ":monitored_session",
        ":session_run_hook",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "ftrl",
    srcs = ["ftrl.py"],
    deps = [
        ":optimizer",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:training_ops_gen",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "gradient_descent",
    srcs = ["gradient_descent.py"],
    visibility = [
        "//machina:internal",
        "//third_party/py/tf_slim/layers:__pkg__",
        "//third_party/py/tf_slim/training:__pkg__",
    ],
    deps = [
        ":optimizer",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:training_ops_gen",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "input",
    srcs = ["input.py"],
    visibility = [
        "//machina:internal",
        "//machina/contrib/training:__pkg__",
    ],
    deps = [
        ":queue_runner",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/layers:layers_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/summary:summary_py",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "momentum",
    srcs = ["momentum.py"],
    visibility = [
        "//machina:internal",
        "//third_party/py/tf_slim/losses:__pkg__",
    ],
    deps = [
        ":optimizer",
        "//machina/python/framework:ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:training_ops_gen",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "moving_averages",
    srcs = ["moving_averages.py"],
    visibility = [
        "//machina:internal",
        "//third_party/py/tf_slim/layers:__pkg__",
    ],
    deps = [
        ":slot_creator",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:reduce_util",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/util:tf_export",
        "//machina/tools/docs:doc_controls",
    ],
)

py_strict_library(
    name = "optimizer",
    srcs = ["optimizer.py"],
    visibility = [
        "//machina:internal",
        "//third_party/py/tf_slim/layers:__pkg__",
        "//third_party/py/tf_slim/training:__pkg__",
    ],
    deps = [
        ":slot_creator",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:distribute_utils",
        "//machina/python/distribute:reduce_util",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:gradients",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/trackable:base",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "proximal_adagrad",
    srcs = ["proximal_adagrad.py"],
    deps = [
        ":optimizer",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:training_ops_gen",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "proximal_gradient_descent",
    srcs = ["proximal_gradient_descent.py"],
    deps = [
        ":optimizer",
        "//machina/python/framework:ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:training_ops_gen",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "quantize_training",
    srcs = ["quantize_training.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python:_pywrap_quantize_training",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "queue_runner_impl",
    srcs = ["queue_runner_impl.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "queue_runner",
    srcs = ["queue_runner.py"],
    visibility = [
        "//machina:internal",
        "//machina/contrib/training:__pkg__",
    ],
    deps = [":queue_runner_impl"],
)

py_strict_library(
    name = "rmsprop",
    srcs = ["rmsprop.py"],
    deps = [
        ":optimizer",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:training_ops_gen",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "session_manager",
    srcs = ["session_manager.py"],
    deps = [
        ":saver",
        "//machina/python/checkpoint:checkpoint_lib",
        "//machina/python/client:session",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "slot_creator",
    srcs = ["slot_creator.py"],
    deps = [
        "//machina/python/compiler/xla/experimental:xla_sharding",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:ref_variable",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
    ],
)

py_strict_library(
    name = "summary_io",
    srcs = ["summary_io.py"],
    deps = [
        "//machina/python/summary:summary_iterator",
        "//machina/python/summary/writer",
        "//machina/python/summary/writer:writer_cache",
        "//machina/python/util:deprecation",
    ],
)

py_strict_library(
    name = "sync_replicas_optimizer",
    srcs = ["sync_replicas_optimizer.py"],
    deps = [
        ":optimizer",
        ":queue_runner",
        ":session_manager",
        ":session_run_hook",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "warm_starting_util",
    srcs = ["warm_starting_util.py"],
    deps = [
        ":checkpoint_ops",
        ":checkpoint_utils",
        ":saver",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
        "//machina/python/training/saving:saveable_object_util",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "server_lib_test",
    size = "small",
    srcs = ["server_lib_test.py"],
    grpc_enabled = True,
    deps = [
        ":input",
        ":queue_runner_impl",
        ":server_lib",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "server_lib_multiple_containers_test",
    size = "small",
    srcs = ["server_lib_multiple_containers_test.py"],
    grpc_enabled = True,
    deps = [
        ":server_lib",
        "//machina/python/client:session",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "server_lib_same_variables_clear_container_test",
    size = "small",
    srcs = ["server_lib_same_variables_clear_container_test.py"],
    grpc_enabled = True,
    deps = [
        ":server_lib",
        "//machina/python/client:session",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "server_lib_same_variables_clear_test",
    size = "small",
    srcs = ["server_lib_same_variables_clear_test.py"],
    grpc_enabled = True,
    deps = [
        ":server_lib",
        "//machina/python/client:session",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "server_lib_same_variables_no_clear_test",
    size = "small",
    srcs = ["server_lib_same_variables_no_clear_test.py"],
    grpc_enabled = True,
    deps = [
        ":server_lib",
        "//machina/python/client:session",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "server_lib_sparse_job_test",
    size = "small",
    srcs = ["server_lib_sparse_job_test.py"],
    grpc_enabled = True,
    deps = [
        ":server_lib",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "localhost_cluster_performance_test",
    size = "medium",
    srcs = [
        "localhost_cluster_performance_test.py",
    ],
    grpc_enabled = True,
    tags = [
        "no_oss",  # Test flaky due to port collisions.
        "oss_serial",
    ],
    deps = [
        ":device_setter",
        "//machina/python:distributed_framework_test_lib",
        "//machina/python/client:session",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "sync_replicas_optimizer_test",
    size = "medium",
    srcs = [
        "sync_replicas_optimizer_test.py",
    ],
    grpc_enabled = True,
    tags = [
        "no_oss",  # Test flaky due to port collisions.
        "oss_serial",
    ],
    deps = [
        ":adam",
        ":gradient_descent",
        ":training_lib",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "evaluation_test",
    size = "small",
    srcs = ["evaluation_test.py"],
    shard_count = 3,
    tags = [
        "manual",
        "notap",  # Disabling until b/33000128 and b/33040312 are fixed.
    ],
    deps = [
        ":basic_session_run_hooks",
        ":evaluation",
        ":gradient_descent",
        ":monitored_session",
        ":saver",
        ":training_lib",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:random_seed",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:metrics",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops/losses",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "py_checkpoint_reader",
    srcs = ["py_checkpoint_reader.py"],
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/util:_pywrap_checkpoint_reader",
        "//machina/python/util:compat",
        "//machina/python/util:tf_export",
    ],
)

tf_proto_library(
    name = "checkpoint_state",
    srcs = ["checkpoint_state.proto"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "checkpoint_state_py_pb2",
#     testonly = 0,
#     deps = [":checkpoint_state"],
# )
# copybara:uncomment_end

py_strict_library(
    name = "checkpoint_management",
    srcs = ["checkpoint_management.py"],
    deps = [
        "//machina/python/checkpoint:checkpoint_management",
        "//machina/python/util:deprecation",
    ],
)

py_strict_library(
    name = "saver",
    srcs = ["saver.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = [
    # "//third_party/py/tf_slim:__subpackages__",
    # "//machina:internal",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":py_checkpoint_reader",
        ":training_util",
        "//machina/core:protos_all_py",
        "//machina/python/checkpoint:checkpoint_management",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:device",
        "//machina/python/framework:errors",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:io_ops_gen",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:pywrap_saved_model",
        "//machina/python/trackable:base",
        "//machina/python/training/saving:saveable_object",
        "//machina/python/training/saving:saveable_object_util",
        "//machina/python/util:compat",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "learning_rate_decay",
    srcs = ["learning_rate_decay.py"],
    deps = ["//machina/python/keras/optimizer_v2:legacy_learning_rate_decay"],
)

py_strict_library(
    name = "saver_test_utils",
    srcs = ["saver_test_utils.py"],
    deps = [
        ":saver",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:lookup_ops_gen",
    ],
)

cuda_py_strict_test(
    name = "saver_test",
    size = "medium",
    srcs = [
        "saver_test.py",
    ],
    tags = ["multi_gpu"],
    deps = [
        ":adam",
        ":gradient_descent",
        ":py_checkpoint_reader",
        ":queue_runner_impl",
        ":saver",
        ":saver_test_utils",
        "//machina/core:protos_all_py",
        "//machina/python/checkpoint:checkpoint_management",
        "//machina/python/client:session",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:iterator_ops",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:function",
        "//machina/python/framework:graph_io",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/saved_model:pywrap_saved_model",
        "//machina/python/summary:summary_py",
        "//machina/python/trackable:base",
        "//machina/python/util:compat",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "saver_large_variable_test",
    size = "medium",
    srcs = ["saver_large_variable_test.py"],
    tags = ["manual"],
    deps = [
        ":saver",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "saver_large_partitioned_variable_test",
    size = "medium",
    srcs = ["saver_large_partitioned_variable_test.py"],
    deps = [
        ":saver",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "basic_session_run_hooks",
    srcs = ["basic_session_run_hooks.py"],
    visibility = [
        "//machina:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":session_run_hook",
        ":summary_io",
        ":training_util",
        "//machina/core:protos_all_py",
        "//machina/python/client:timeline",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "session_run_hook",
    srcs = ["session_run_hook.py"],
    visibility = [
        "//machina:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = ["//machina/python/util:tf_export"],
)

py_strict_library(
    name = "supervisor",
    srcs = ["supervisor.py"],
    deps = [
        ":coordinator",
        ":saver",
        ":session_manager",
        ":training_util",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
        "//machina/python/summary:summary_py",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "supervisor_test",
    size = "small",
    srcs = ["supervisor_test.py"],
    grpc_enabled = True,
    tags = ["no_windows"],
    deps = [
        ":input",
        ":saver",
        ":server_lib",
        ":session_manager",
        ":supervisor",
        "//machina/core:protos_all_py",
        "//machina/python/checkpoint:checkpoint_management",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/summary:summary_iterator",
        "//machina/python/summary:summary_py",
        "//machina/python/summary/writer",
    ],
)

py_strict_library(
    name = "server_lib",
    srcs = ["server_lib.py"],
    visibility = [
        "//smartass/brain/ops:__subpackages__",
        "//machina:internal",
    ],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/framework:errors",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "training_util",
    srcs = ["training_util.py"],
    visibility = [
        "//machina:internal",
        "//third_party/cloud_tpu/convergence_tools:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:graph_io",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:cond",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "training_util_test",
    size = "small",
    srcs = ["training_util_test.py"],
    deps = [
        ":monitored_session",
        ":training_util",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "adam_test",
    size = "medium",
    srcs = ["adam_test.py"],
    tags = ["no_rocm"],
    deps = [
        ":adam",
        "//machina/python/client:session",
        "//machina/python/compiler/xla/experimental:xla_sharding",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:ref_variable",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "moving_averages_test",
    size = "small",
    srcs = [
        "moving_averages_test.py",
    ],
    tags = [
        "no_windows",  # b/139083295: bfloat16 tests fail on Windows
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":moving_averages",
        ":saver",
        "//machina/python/compiler/xla/experimental:xla_sharding",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:state_ops_gen",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

TRAINING_TEST_DEPS = [
    "//third_party/py/numpy",
    "//machina/core:protos_all_py",
    "//machina/python/ops:array_ops",
    "//machina/python/client:client",
    "//machina/python/platform:client_testlib",
    "//machina/python/ops:cond",
    "//machina/python/ops:control_flow_ops",
    "//machina/python/ops:data_flow_ops",
    "//machina/python/ops:data_flow_ops_gen",
    "//machina/python/ops:embedding_ops",
    "//machina/python/framework:errors",
    "//machina/python/framework:for_generated_wrappers",
    "//machina/python/framework:test_lib",
    "//machina/python/ops:custom_gradient",
    "//machina/python/ops:gradients",
    "//machina/python/ops:lookup_ops",
    "//machina/python/ops:math_ops",
    "//machina/python/ops:nn_grad",
    "//machina/python/ops:nn_ops",
    "//machina/python/ops:partitioned_variables",
    "//machina/python/platform:test",
    "//machina/python:pywrap_machina",
    "//machina/python/ops:random_ops",
    "//machina/python/ops:resource_variable_ops",
    "//machina/python/ops:resources",
    "//machina/python/ops:sparse_ops",
    "//machina/python/ops:state_ops",
    "//machina/python/ops:state_ops_gen",
    "//machina/python/ops:variable_scope",
    "//machina/python/ops:variables",
    "//machina/python/distribute:cross_device_ops",
    "//machina/python/distribute:distribute_utils",
    "//machina/python/distribute:mirrored_strategy",
]

cuda_py_strict_test(
    name = "adadelta_test",
    size = "medium",
    srcs = ["adadelta_test.py"],
    deps = [
        ":adadelta",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "adagrad_da_test",
    size = "medium",
    srcs = ["adagrad_da_test.py"],
    deps = [
        ":adagrad_da",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "adagrad_test",
    size = "medium",
    srcs = ["adagrad_test.py"],
    deps = [
        ":adagrad",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "basic_loops_test",
    size = "medium",
    srcs = ["basic_loops_test.py"],
    deps = [
        ":basic_loops",
        ":supervisor",
        "//machina/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "coordinator_test",
    size = "medium",
    srcs = ["coordinator_test.py"],
    deps = [
        ":coordinator",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "device_setter_test",
    size = "medium",
    srcs = ["device_setter_test.py"],
    deps = [
        ":device_setter",
        ":server_lib",
        "//machina/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "ftrl_test",
    size = "medium",
    srcs = ["ftrl_test.py"],
    deps = [
        ":adagrad",
        ":ftrl",
        ":gradient_descent",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "gradient_descent_test",
    size = "medium",
    srcs = ["gradient_descent_test.py"],
    deps = [
        ":gradient_descent",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "momentum_test",
    size = "medium",
    srcs = ["momentum_test.py"],
    deps = [
        ":momentum",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "optimizer_test",
    size = "medium",
    srcs = ["optimizer_test.py"],
    deps = [
        ":adam",
        ":gradient_descent",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:clip_ops",
        "//machina/python/ops:gradients_util",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "proximal_adagrad_test",
    size = "medium",
    srcs = ["proximal_adagrad_test.py"],
    deps = [
        ":adagrad",
        ":proximal_adagrad",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "proximal_gradient_descent_test",
    size = "medium",
    srcs = ["proximal_gradient_descent_test.py"],
    deps = [
        ":gradient_descent",
        ":proximal_gradient_descent",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "quantize_training_test",
    size = "medium",
    srcs = ["quantize_training_test.py"],
    deps = [
        ":quantize_training",
        ":saver",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/ops:variable_v1",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "queue_runner_test",
    size = "medium",
    srcs = ["queue_runner_test.py"],
    deps = [
        ":coordinator",
        ":monitored_session",
        ":queue_runner_impl",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:variable_v1",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "rmsprop_test",
    size = "medium",
    srcs = ["rmsprop_test.py"],
    deps = [
        ":rmsprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "slot_creator_test",
    size = "medium",
    srcs = ["slot_creator_test.py"],
    deps = [
        ":slot_creator",
        "//machina/python/compiler/xla/experimental:xla_sharding",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:variable_v1",
        "@local_xla//xla:xla_data_proto_py",
    ] + TRAINING_TEST_DEPS,
)

cuda_py_strict_test(
    name = "training_ops_test",
    size = "medium",
    srcs = ["training_ops_test.py"],
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:training_ops_gen",
        "//machina/python/ops:variable_v1",
    ] + TRAINING_TEST_DEPS,
)

test_suite(
    name = "training_tests",
    tests = [
        "adadelta_test",
        "adagrad_da_test",
        "adagrad_test",
        "basic_loops_test",
        "coordinator_test",
        "device_setter_test",
        "ftrl_test",
        "gradient_descent_test",
        "momentum_test",
        "optimizer_test",
        "proximal_adagrad_test",
        "proximal_gradient_descent_test",
        "quantize_training_test",
        "queue_runner_test",
        "rmsprop_test",
        "slot_creator_test",
        "training_ops_test",
    ],
)

distribute_py_strict_test(
    name = "training_ops_mlir_test",
    srcs = [
        "training_ops_test.py",
    ],
    disable_mlir_bridge = False,
    main = "training_ops_test.py",
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:training_ops_gen",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/platform:test",
    ],
)

cuda_py_strict_test(
    name = "session_manager_test",
    size = "medium",  # TODO(irving): Can this be made small?
    srcs = ["session_manager_test.py"],
    grpc_enabled = True,
    main = "session_manager_test.py",
    deps = [
        ":saver",
        ":server_lib",
        ":session_manager",
        "//machina/python/checkpoint:checkpoint_management",
        "//machina/python/client:session",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:resource_variables_toggle",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
    ],
)

tf_py_strict_test(
    name = "basic_session_run_hooks_test",
    size = "medium",
    srcs = ["basic_session_run_hooks_test.py"],
    tags = [
        "no_pip",  # Relies on contrib
        "no_windows",
        "notsan",  # intermittent races on a few percent of runs
    ],
    deps = [
        ":basic_session_run_hooks",
        ":checkpoint_utils",
        ":monitored_session",
        ":session_run_hook",
        ":training_util",
        "//machina/python/client:session",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/summary:summary_py",
        "//machina/python/summary/writer:fake_summary_writer",
        "//machina/python/summary/writer:writer_cache",
    ],
)

tf_py_strict_test(
    name = "checkpoint_utils_iterator_test",
    size = "small",
    srcs = ["checkpoint_utils_iterator_test.py"],
    deps = [
        ":checkpoint_utils",
        ":saver",
        "//machina/core:protos_all_py",
        "//machina/python/checkpoint",
        "//machina/python/client:session",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
    ],
)

tf_py_strict_test(
    name = "checkpoint_utils_test",
    size = "small",
    srcs = ["checkpoint_utils_test.py"],
    tags = [
        "manual",
        "no_cuda_on_cpu_tap",
        "no_oss",
        "no_windows",
        "notap",
    ],
    deps = [
        ":checkpoint_utils",
        ":saver",
        "//machina/core:protos_all_py",
        "//machina/python/checkpoint",
        "//machina/python/client:session",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
    ],
)

tf_py_strict_test(
    name = "checkpoint_ops_test",
    size = "small",
    srcs = ["checkpoint_ops_test.py"],
    deps = [
        ":checkpoint_ops",
        ":saver",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "warm_starting_util_test",
    size = "medium",
    srcs = ["warm_starting_util_test.py"],
    deps = [
        ":checkpoint_utils",
        ":saver",
        ":warm_starting_util",
        "//machina/python/checkpoint",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "monitored_session",
    srcs = ["monitored_session.py"],
    visibility = [
        "//machina:internal",
        "//third_party/py/tf_slim/training:__pkg__",
    ],
    deps = [
        ":basic_session_run_hooks",
        ":coordinator",
        ":queue_runner",
        ":saver",
        ":session_manager",
        ":session_run_hook",
        "//machina/core:protos_all_py",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint:graph_view",
        "//machina/python/distribute:distribute_coordinator_context",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:resources",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
        "//machina/python/summary:summary_py",
        "//machina/python/util:function_utils",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "monitored_session_test",
    size = "medium",
    srcs = ["monitored_session_test.py"],
    tags = [
        "no_pip",
        "notsan",  # b/67945581
    ],
    deps = [
        ":basic_session_run_hooks",
        ":coordinator",
        ":monitored_session",
        ":saver",
        ":session_run_hook",
        ":summary_io",
        ":training_util",
        "//machina/core:protos_all_py",
        "//machina/python/checkpoint:checkpoint_management",
        "//machina/python/client:session",
        "//machina/python/distribute:collective_all_reduce_strategy",
        "//machina/python/distribute:distribute_coordinator",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:builder",
        "//machina/python/saved_model:load",
        "//machina/python/summary:summary_py",
    ],
)

tf_py_strict_test(
    name = "input_test",
    size = "medium",
    srcs = ["input_test.py"],
    deps = [
        ":coordinator",
        ":input",
        ":queue_runner_impl",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
        "//third_party/py/numpy",
    ],
)
