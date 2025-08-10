load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "tf_py_strict_test")
load("//machina/core/platform:distribute.bzl", "distribute_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "cluster_coordinator",
    srcs = ["cluster_coordinator.py"],
    deps = [
        ":coordinator_context",
        ":metric_utils",
        ":remote_value",
        ":utils",
        ":values",
        ":watchdog",
        "//machina/python/eager:cancellation",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:executor",
        "//machina/python/eager:function",
        "//machina/python/framework:errors",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "coordinator_context",
    srcs = [
        "coordinator_context.py",
    ],
    deps = [
        ":remote_value",
        "//machina/core:protos_all_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/util:compat",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "remote_value",
    srcs = ["remote_value.py"],
    deps = ["//machina/python/util:tf_export"],
)

py_strict_library(
    name = "values",
    srcs = ["values.py"],
    deps = [
        ":remote_value",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/distribute:input_lib",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:function",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:type_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:dataset_ops_gen",
        "//machina/python/ops:experimental_dataset_ops_gen",
        "//machina/python/ops:variable_scope",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

distribute_py_strict_test(
    name = "cluster_coordinator_test",
    srcs = ["cluster_coordinator_test.py"],
    shard_count = 50,
    tags = [
        # "multi_gpu", # TODO(b/287692888): re-enable once the 2gpu test passes.
        "no_oss",  # TODO(b/214432000): Very flaky under Docker
        "no_pip",
        "noasan",  # TODO(b/171040359): Flaky timeout, even if maximum shards
        "notpu",
        "notsan",  # TODO(b/171040359): Flaky timeout, even if maximum shards
    ],
    xla_enable_strict_auto_jit = False,  # TODO(b/291174864)
    xla_tags = [
        "no_cuda_asan",  # Race condition on async test
    ],
    deps = [
        ":cluster_coordinator",
        ":coordinator_context",
        ":remote_value",
        "//machina/python/compat:v2_compat",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:distribute_utils",
        "//machina/python/distribute:input_lib",
        "//machina/python/distribute:multi_worker_test_base",
        "//machina/python/distribute:parameter_server_strategy_v2",
        "//machina/python/distribute:sharded_variable",
        "//machina/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//machina/python/eager:cancellation",
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:embedding_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:coordinator",
        "//machina/python/training:server_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "fault_tolerance_test_base",
    srcs = ["fault_tolerance_test_base.py"],
    deps = [
        ":cluster_coordinator",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute:multi_worker_test_base",
        "//machina/python/distribute:parameter_server_strategy_v2",
        "//machina/python/distribute:test_util",
        "//machina/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:coordinator",
        "//machina/python/training:server_lib",
    ],
)

tf_py_strict_test(
    name = "fault_tolerance_test",
    srcs = ["fault_tolerance_test.py"],
    shard_count = 41,  # = number of tests, so one shard = one test
    tags = [
        "no_oss",  # TODO(b/219580021)
        "noasan",  # Multi-process runner does not work with test sanitizers
        "nomac",  # TODO(b/177065434)
        "nomsan",
    ],
    deps = [
        ":cluster_coordinator",
        ":fault_tolerance_test_base",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute:multi_process_runner",
        "//machina/python/distribute:multi_worker_test_base",
        "//machina/python/distribute:parameter_server_strategy_v2",
        "//machina/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//machina/python/eager:test",
        "//machina/python/training:coordinator",
        "//machina/python/training:server_lib",
    ],
)

tf_py_strict_test(
    name = "fault_tolerance_coordination_service_test",
    srcs = ["fault_tolerance_coordination_service_test.py"],
    shard_count = 41,
    tags = [
        # Inherit tags from fault_tolerance_test
        "no_oss",  # TODO(b/219580021)
        "noasan",  # Multi-process runner does not work with test sanitizers
        "nomac",  # TODO(b/177065434)
        "nomsan",  # TODO(b/290211396)
        "notap",
    ],
    deps = [
        ":cluster_coordinator",
        ":fault_tolerance_test_base",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute:multi_process_runner",
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
    ],
)

py_strict_library(
    name = "metric_utils",
    srcs = ["metric_utils.py"],
    deps = [
        "//machina/python/eager:monitoring",
        "//machina/python/util:tf_contextlib",
    ],
)

tf_py_strict_test(
    name = "metric_utils_test",
    srcs = ["metric_utils_test.py"],
    shard_count = 3,
    deps = [
        ":cluster_coordinator",
        ":metric_utils",
        "//machina/python/distribute:multi_process_runner",
        "//machina/python/distribute:multi_worker_test_base",
        "//machina/python/distribute:parameter_server_strategy_v2",
        "//machina/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/training:coordinator",
        "//machina/python/training:server_lib",
    ],
)

py_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//machina/python/platform:tf_logging",
        "//machina/python/training:server_lib",
    ],
)

py_strict_library(
    name = "remote_eager_lib",
    visibility = ["//visibility:public"],
)

py_strict_library(
    name = "watchdog",
    srcs = ["watchdog.py"],
    deps = ["@absl_py//absl/logging"],
)

tf_py_strict_test(
    name = "watchdog_test",
    srcs = ["watchdog_test.py"],
    tags = [
        "nomac",  # TODO(b/239433962)
    ],
    deps = [
        ":watchdog",
        "//machina/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "get_task_states_test",
    srcs = ["get_task_states_test.py"],
    shard_count = 3,
    tags = [
        "no_oss",  # TODO(b/219580021)
        "nomac",  # TODO(b/177065434)
    ],
    deps = [
        ":cluster_coordinator",
        ":utils",
        "//machina/core:protos_all_py",
        "//machina/python/compat:v2_compat",
        "//machina/python/distribute:multi_process_runner",
        "//machina/python/distribute:multi_worker_test_base",
        "//machina/python/distribute:parameter_server_strategy_v2",
        "//machina/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//machina/python/eager:context",
        "//machina/python/eager:test",
        "//machina/python/framework:errors",
        "//machina/python/training:server_lib",
    ],
)
