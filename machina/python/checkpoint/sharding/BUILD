# Description:
#   Utilities for sharding object-based checkpoints.

load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:machina.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "sharding_policies",
    srcs = ["sharding_policies.py"],
    deps = [
        ":sharding_util",
        "//machina/python/eager:context",
        "//machina/python/framework:device",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:variables",
        "//machina/python/trackable:base",
        "//machina/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "sharding_policies_test",
    srcs = ["sharding_policies_test.py"],
    deps = [
        ":sharding_policies",
        ":sharding_util",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint:checkpoint_options",
        "//machina/python/checkpoint:graph_view",
        "//machina/python/eager:def_function",
        "//machina/python/eager:remote",
        "//machina/python/eager:test",
        "//machina/python/framework:device",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/module",
        "//machina/python/ops:variables",
        "//machina/python/platform:gfile",
        "//machina/python/training:server_lib",
        "//machina/python/training/saving:saveable_object",
        "//machina/python/training/saving:saveable_object_util",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "sharding_util",
    srcs = ["sharding_util.py"],
    deps = [
        "//machina/python/framework:device",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:variables",
        "//machina/python/trackable:base",
        "//machina/python/training/saving:saveable_object",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "sharding_util_test",
    srcs = ["sharding_util_test.py"],
    deps = [
        ":sharding_policies",
        ":sharding_util",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint:graph_view",
        "//machina/python/eager:remote",
        "//machina/python/eager:test",
        "//machina/python/framework:device",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/training:server_lib",
        "//machina/python/training/saving:saveable_object",
        "//machina/python/training/saving:saveable_object_util",
    ],
)
