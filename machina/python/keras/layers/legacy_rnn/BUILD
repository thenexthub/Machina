# Description:
#   Contains the legacy TF RNN APIs (internal TensorFlow version).

load("//machina:py.default.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:__subpackages__",
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
    name = "rnn_cell_impl",
    srcs = ["rnn_cell_impl.py"],
    srcs_version = "PY3",
    deps = [
        ":rnn_cell_wrapper_impl",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/keras:activations",
        "//machina/python/keras/engine:input_spec",
        "//machina/python/keras/initializers",
        "//machina/python/keras/legacy_tf_layers:layers_base",
        "//machina/python/keras/saving",
        "//machina/python/keras/utils:tf_utils",
        "//machina/python/layers:layers_base",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:clip_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
        "//machina/python/trackable:base",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

py_library(
    name = "rnn_cell_wrapper_impl",
    srcs = ["rnn_cell_wrapper_impl.py"],
    srcs_version = "PY3",
    deps = [
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/keras/utils:generic_utils",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/util:nest",
    ],
)
