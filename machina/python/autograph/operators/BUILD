load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "py_builtins",
    srcs = ["py_builtins.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/utils:tensors",
        "//machina/python/autograph/utils:type_registry",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:parsing_ops_gen",
        "//machina/python/ops:string_ops_gen",
    ],
)

py_strict_library(
    name = "exceptions",
    srcs = ["exceptions.py"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/util:tf_inspect",
    ],
)

py_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        ":conditional_expressions",
        ":control_flow",
        ":data_structures",
        ":exceptions",
        ":logical",
        ":py_builtins",
        ":slices",
        ":variables",
    ],
)

py_strict_library(
    name = "logical",
    srcs = ["logical.py"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:cond",
        "//machina/python/ops:math_ops_gen",
    ],
)

py_strict_library(
    name = "variables",
    srcs = ["variables.py"],
    visibility = ["//machina:__subpackages__"],
)

py_strict_library(
    name = "data_structures",
    srcs = ["data_structures.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:tensor_array_ops",
    ],
)

py_strict_library(
    name = "conditional_expressions",
    srcs = ["conditional_expressions.py"],
    visibility = ["//visibility:private"],
    deps = [
        ":control_flow",
        "//machina/python/autograph/utils:tensors",
        "//machina/python/ops:cond",
    ],
)

py_strict_library(
    name = "control_flow",
    srcs = ["control_flow.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        ":py_builtins",
        ":variables",
        "//machina/python/autograph/utils:ag_logging",
        "//machina/python/autograph/utils:misc",
        "//machina/python/autograph/utils:tensors",
        "//machina/python/autograph/utils:type_registry",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:control_flow_util",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/types:distribute",
        "//machina/python/util:nest",
        "//machina/python/util:variable_utils",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "slices",
    srcs = ["slices.py"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:string_ops_gen",
        "//machina/python/ops:tensor_array_ops",
    ],
)

py_strict_test(
    name = "data_structures_test",
    srcs = ["data_structures_test.py"],
    deps = [
        ":data_structures",
        #internal proto upb dep
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "conditional_expressions_test",
    srcs = ["conditional_expressions_test.py"],
    deps = [
        ":conditional_expressions",
        #internal proto upb dep
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "control_flow_test",
    srcs = ["control_flow_test.py"],
    tags = [
        "no_gpu",  # b/127001953
    ],
    deps = [
        ":control_flow",
        ":variables",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina/python/autograph/utils:ag_logging",
        "//machina/python/autograph/utils:testing",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:random_ops",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "exceptions_test",
    srcs = ["exceptions_test.py"],
    deps = [
        ":exceptions",
        #internal proto upb dep
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "logical_test",
    srcs = ["logical_test.py"],
    deps = [
        ":logical",
        #internal proto upb dep
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "py_builtins_test",
    srcs = ["py_builtins_test.py"],
    deps = [
        ":data_structures",
        ":py_builtins",
        #internal proto upb dep
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/core:function_wrappers",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "slices_test",
    srcs = ["slices_test.py"],
    deps = [
        ":slices",
        #internal proto upb dep
        "//machina/python/framework:constant_op",
        "//machina/python/ops:list_ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "variables_test",
    srcs = ["variables_test.py"],
    deps = [
        ":variables",
        #internal proto upb dep
        "//machina/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "dispatch_context",
    srcs = ["dispatch_context.py"],
)
