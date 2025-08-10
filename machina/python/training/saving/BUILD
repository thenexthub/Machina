# Description:
#   Low-level utilities for reading and writing checkpoints.

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
    name = "checkpoint_options",
    srcs = ["checkpoint_options.py"],
    deps = [
        "//machina/python/checkpoint:checkpoint_options",
        "//machina/python/util:deprecation",
    ],
)

py_strict_library(
    name = "functional_saver",
    srcs = ["functional_saver.py"],
    deps = [
        "//machina/python/checkpoint:functional_saver",
        "//machina/python/util:deprecation",
    ],
)

py_strict_library(
    name = "saveable_object",
    srcs = ["saveable_object.py"],
)

py_strict_library(
    name = "saveable_object_util",
    srcs = ["saveable_object_util.py"],
    deps = [
        ":saveable_object",
        "//machina/python/checkpoint:saveable_compat",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:device",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops_gen",
        "//machina/python/ops:ref_variable",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
        "//machina/python/trackable:base",
        "//machina/python/trackable:base_delegate",
        "//machina/python/trackable:python_state",
        "//machina/python/trackable:trackable_utils",
        "//machina/python/types:core",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "trace_saveable_util",
    srcs = ["trace_saveable_util.py"],
    deps = [
        ":saveable_object",
        ":saveable_object_util",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:type_spec",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/util:nest",
    ],
)

tf_py_strict_test(
    name = "saveable_object_util_test",
    srcs = ["saveable_object_util_test.py"],
    deps = [
        ":saveable_object",
        ":saveable_object_util",
        "//machina/python/checkpoint",
        "//machina/python/eager:context",
        "//machina/python/eager:test",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:resource_variable_ops_gen",
        "//machina/python/ops:variables",
        "//machina/python/trackable:base",
        "//machina/python/trackable:resource",
    ],
)
