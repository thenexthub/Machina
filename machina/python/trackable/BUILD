# Description:
#   Trackable class and subclass definitions.

load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "trackable",
    visibility = [
        "//machina:internal",
        "//third_party/py/tf_agents:__subpackages__",
    ],
    deps = [
        ":asset",
        ":autotrackable",
        ":base",
        ":base_delegate",
        ":constants",
        ":converter",
        ":data_structures",
        ":layer_utils",
        ":python_state",
        ":resource",
        ":trackable_init",
        ":trackable_utils",
    ],
)

py_strict_library(
    name = "trackable_init",
    srcs = ["__init__.py"],
)

py_strict_library(
    name = "base",
    srcs = ["base.py"],
    deps = [
        ":constants",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:control_flow_ops_gen",
        "//machina/python/training/saving:saveable_object",
        "//machina/python/util:tf_contextlib",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "base_test",
    srcs = ["base_test.py"],
    deps = [
        ":base",
        "//machina/python/checkpoint",
        "//machina/python/framework:ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "constants",
    srcs = ["constants.py"],
)

py_strict_library(
    name = "converter",
    srcs = ["converter.py"],
    deps = [
        ":base",
        ":data_structures",
        "//machina/python/eager/polymorphic_function:saved_model_utils",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:resource_variable_ops",
    ],
)

py_strict_library(
    name = "trackable_utils",
    srcs = ["trackable_utils.py"],
)

tf_py_strict_test(
    name = "trackable_utils_test",
    srcs = ["trackable_utils_test.py"],
    deps = [
        ":trackable_utils",
        "//machina/python/eager:test",
    ],
)

py_strict_library(
    name = "base_delegate",
    srcs = ["base_delegate.py"],
    deps = [
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "base_delegate_test",
    srcs = ["base_delegate_test.py"],
    deps = [
        ":base",
        ":base_delegate",
        "//machina/python:extra_py_tests_deps",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint:checkpoint_options",
        "//machina/python/eager:test",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:variables",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:save",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "asset",
    srcs = ["asset.py"],
    deps = [
        ":base",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion_registry",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/saved_model:path_helpers",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "autotrackable",
    srcs = ["autotrackable.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":base",
        ":data_structures",
        "//machina/python/eager:def_function",
        "//machina/python/eager:function",
        "//machina/python/types:core",
        "//machina/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "autotrackable_test",
    srcs = ["autotrackable_test.py"],
    deps = [
        ":autotrackable",
        ":data_structures",
        "//machina/python/checkpoint",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:nest",
    ],
)

py_strict_library(
    name = "resource",
    srcs = ["resource.py"],
    visibility = [
        "//intelligence/climate_foundations/vlm/export:__subpackages__",
        "//machina:internal",
    ],
    deps = [
        ":base",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/util:tf_contextlib",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "resource_test",
    srcs = ["resource_test.py"],
    deps = [
        ":resource",
        "//machina/python/eager:context",
        "//machina/python/eager:wrap_function",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "layer_utils",
    srcs = ["layer_utils.py"],
    deps = ["//machina/python/util:object_identity"],
)

py_strict_library(
    name = "data_structures",
    srcs = ["data_structures.py"],
    deps = [
        ":base",
        ":layer_utils",
        "//machina/python/eager:def_function",
        "//machina/python/eager:function",
        "//machina/python/ops:variables",
        "//machina/python/util:compat",
        "//machina/python/util:tf_export",
        "@pypi_wrapt//:pkg",
    ],
)

tf_py_strict_test(
    name = "data_structures_test",
    srcs = ["data_structures_test.py"],
    tags = [
        "no_oss",  # Keras is not available in OSS test
        "no_windows",
        "nomac",
    ],
    deps = [
        ":autotrackable",
        ":data_structures",
        "//machina/python/checkpoint",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:tensor_shape",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/util:nest",
        "//machina/python/util:serialization",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "python_state",
    srcs = ["python_state.py"],
    deps = [
        ":base",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "python_state_test",
    srcs = ["python_state_test.py"],
    deps = [
        ":python_state",
        "//machina/python/checkpoint",
        "//machina/python/client:session",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/module",
        "//machina/python/platform:client_testlib",
    ],
)
