# Description:
#   Contains the Keras OptimizerV2 API (internal TensorFlow version).

load("//machina:py.default.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//machina/python:__pkg__",
        "//machina/python/distribute:__pkg__",
        "//machina/python/keras:__subpackages__",
        "//machina/python/tpu/tests:__pkg__",
        "//machina/python/trackable:__pkg__",
        "//machina/python/training:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//machina/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "optimizer_v2",
    srcs = [
        "adadelta.py",
        "adagrad.py",
        "adam.py",
        "adamax.py",
        "ftrl.py",
        "gradient_descent.py",
        "nadam.py",
        "optimizer_v2.py",
        "rmsprop.py",
        "utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":learning_rate_schedule",
        "//machina/python/distribute:central_storage_strategy",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:parameter_server_strategy",
        "//machina/python/distribute:parameter_server_strategy_v2",
        "//machina/python/distribute:reduce_util",
        "//machina/python/distribute:values",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/keras:backend",
        "//machina/python/keras:backend_config",
        "//machina/python/keras/engine:base_layer_utils",
        "//machina/python/keras/initializers",
        "//machina/python/keras/utils:layer_utils",
        "//machina/python/keras/utils:tf_utils",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
    ],
)

py_library(
    name = "learning_rate_schedule",
    srcs = [
        "learning_rate_schedule.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/keras/utils:generic_utils",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_case",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
    ],
)

py_library(
    name = "legacy_learning_rate_decay",
    srcs = ["legacy_learning_rate_decay.py"],
    srcs_version = "PY3",
    deps = [
        ":learning_rate_schedule",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/ops:math_ops",
        "//machina/python/util:tf_export",
    ],
)
