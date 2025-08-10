# Description:
#   keras/distribute package is intended to serve as the centralized place for things
#   related to dist-strat used by Keras..

load("//machina:py.default.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    # TODO(scottzhu): Remove this deps when distribute test are converted to integration test.
    default_visibility = [
        "//machina:internal",
        "//machina/python/distribute:__pkg__",
        "//machina/python/keras:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//machina/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "distribute",
    srcs = [
        "__init__.py",
        "distributed_training_utils.py",
        "distributed_training_utils_v1.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":distribute_coordinator_utils",
        "//machina/python/data",
        "//machina/python/distribute:distribute_coordinator",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:input_lib",
        "//machina/python/distribute:one_device_strategy",
        "//machina/python/distribute:reduce_util",
        "//machina/python/distribute:values",
        "//machina/python/eager:def_function",
        "//machina/python/keras:backend",
        "//machina/python/keras:callbacks",
        "//machina/python/keras:callbacks_v1",
        "//machina/python/keras:constraints",
        "//machina/python/keras:losses",
        "//machina/python/keras:optimizers",
        "//machina/python/keras:regularizers",
        "//machina/python/keras/initializers",
        "//machina/python/keras/mixed_precision:policy",
        "//machina/python/keras/utils:engine_utils",
        "//machina/python/keras/utils:mode_keys",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/python/trackable:data_structures",
        "//machina/tools/docs:doc_controls",
    ],
)

py_library(
    name = "optimizer_combinations",
    srcs = ["optimizer_combinations.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/distribute:combinations",
        "//machina/python/distribute:strategy_combinations",
        "//machina/python/keras/optimizer_v2",
        "//machina/python/training",
    ],
)

py_library(
    name = "worker_training_state",
    srcs = [
        "worker_training_state.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":distributed_file_utils",
        "//machina/python/checkpoint",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/keras:backend",
        "//machina/python/keras/utils:mode_keys",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:variables",
        "//machina/python/training:checkpoint_management",
    ],
)

py_library(
    name = "distributed_file_utils",
    srcs = [
        "distributed_file_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//machina/python/distribute:distribute_lib",
        "//machina/python/lib/io:file_io",
    ],
)

py_library(
    name = "distribute_coordinator_utils",
    srcs = [
        "distribute_coordinator_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/distribute:distribute_coordinator_context",
        "//machina/python/distribute:multi_worker_util",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:server_lib",
    ],
)
