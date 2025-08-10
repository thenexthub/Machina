# Description:
#   Contains the Keras Utilities (internal TensorFlow version).

load("//machina:py.default.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//machina/python/feature_column:__pkg__",
        "//machina/python/keras:__subpackages__",
        "//machina/tools/pip_package:__pkg__",
        "//machina/tools/pip_package:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//machina/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "utils",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":all_utils",
    ],
)

py_library(
    name = "all_utils",
    srcs = [
        "all_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":control_flow_util",
        ":engine_utils",
        ":generic_utils",
        ":layer_utils",
        ":np_utils",
        ":vis_utils",
    ],
)

py_library(
    name = "control_flow_util",
    srcs = ["control_flow_util.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "data_utils",
    srcs = ["data_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":generic_utils",
        ":io_utils",
        ":tf_inspect",
        "//machina/python/framework:tensor",
    ],
)

py_library(
    name = "engine_utils",
    srcs = [
        "conv_utils.py",
        "losses_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":data_utils",
        ":io_utils",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/keras:backend",
    ],
)

py_library(
    name = "io_utils",
    srcs = ["io_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "tf_utils",
    srcs = ["tf_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":object_identity",
        "//machina/python/distribute/coordinator:cluster_coordinator",
        "//machina/python/eager:context",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:variables",
        "//machina/python/util:nest",
    ],
)

py_library(
    name = "generic_utils",
    srcs = [
        "generic_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":tf_contextlib",
        ":tf_inspect",
        "//machina/python/util:nest",
        "//machina/python/util:tf_decorator_py",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "mode_keys",
    srcs = [
        "mode_keys.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//machina/python/saved_model/model_utils:mode_keys",
    ],
)

py_library(
    name = "layer_utils",
    srcs = [
        "kernelized_utils.py",
        "layer_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":engine_utils",
        "//machina/python/keras:backend",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "metrics_utils",
    srcs = [
        "metrics_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":engine_utils",
        ":generic_utils",
        ":tf_utils",
        "//machina/python/distribute",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:weights_broadcast_ops",
        "//machina/python/ops/losses",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/ops/ragged:ragged_util",
        "//machina/python/tpu:tpu_lib",
        "//machina/python/util:tf_decorator_py",
    ],
)

py_library(
    name = "version_utils",
    srcs = [
        "version_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//machina/python/framework:ops",
    ],
)

py_library(
    name = "np_utils",
    srcs = [
        "np_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "object_identity",
    srcs = ["object_identity.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "tf_contextlib",
    srcs = ["tf_contextlib.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/util:tf_decorator_py",
    ],
)

py_library(
    name = "tf_inspect",
    srcs = ["tf_inspect.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/util:tf_decorator_py",
    ],
)

py_library(
    name = "vis_utils",
    srcs = [
        "vis_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//machina/python/util:nest",
    ],
)

py_library(
    name = "dataset_creator",
    srcs = [
        "dataset_creator.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//machina/python/types:data",
        "//machina/python/util:tf_export",
    ],
)
