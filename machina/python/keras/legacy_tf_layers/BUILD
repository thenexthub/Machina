# Description:
#   Contains the legacy TF layers (internal TensorFlow version).

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
    name = "layers_base",
    srcs = [
        "__init__.py",
        "base.py",
        "variable_scope_shim.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/keras:backend",
        "//machina/python/keras/engine:base_layer",
        "//machina/python/keras/mixed_precision:policy",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/trackable:base",
        "//machina/python/util:nest",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
    ],
)

py_library(
    name = "convolutional",
    srcs = ["convolutional.py"],
    srcs_version = "PY3",
    deps = [
        ":layers_base",
        "//machina/python/keras/layers",
        "//machina/python/ops:init_ops",
        "//machina/python/util:tf_export",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    deps = [
        ":layers_base",
        "//machina/python/keras/layers",
        "//machina/python/ops:init_ops",
        "//machina/python/util:tf_export",
    ],
)

py_library(
    name = "pooling",
    srcs = ["pooling.py"],
    srcs_version = "PY3",
    deps = [
        ":layers_base",
        "//machina/python/keras/layers",
        "//machina/python/util:tf_export",
    ],
)
