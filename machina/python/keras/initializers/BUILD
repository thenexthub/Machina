# Description:
#   Contains the Keras initializer API (internal TensorFlow version).

load("//machina:py.default.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/python/keras:__subpackages__"],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = [
        "//machina/python/keras/google/private_tf_api_test:__pkg__",
    ],
)

py_library(
    name = "initializers",
    srcs = [
        "__init__.py",
        "initializers_v1.py",
        "initializers_v2.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//machina/python:tf2",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/keras:backend",
        "//machina/python/keras/utils:generic_utils",
        "//machina/python/keras/utils:tf_inspect",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/util:tf_export",
    ],
)
