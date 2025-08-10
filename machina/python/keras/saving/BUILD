# Description:
#   Contains the Keras save model API (internal TensorFlow version).

load("//machina:py.default.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//machina/python/distribute:__pkg__",
        "//machina/python/keras:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob([
        "*.py",
        "**/*.py",
    ]),
    visibility = ["//machina/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "saving",
    srcs = [
        "__init__.py",
        "hdf5_format.py",
        "model_config.py",
        "save.py",
        "saved_model_experimental.py",
        "saving_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//machina/python/checkpoint:graph_view",
        "//machina/python/eager:def_function",
        "//machina/python/framework:tensor_spec",
        "//machina/python/keras:backend",
        "//machina/python/keras:losses",
        "//machina/python/keras:optimizers",
        "//machina/python/keras:regularizers",
        "//machina/python/keras/engine:input_spec",
        "//machina/python/keras/mixed_precision:autocast_variable",
        "//machina/python/keras/protobuf:saved_metadata_proto_py",
        "//machina/python/keras/saving/saved_model",
        "//machina/python/keras/saving/utils_v1",
        "//machina/python/keras/utils:engine_utils",
        "//machina/python/keras/utils:metrics_utils",
        "//machina/python/keras/utils:mode_keys",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model",
        "//machina/python/training:saver",
        "@pypi_h5py//:pkg",
    ],
)
