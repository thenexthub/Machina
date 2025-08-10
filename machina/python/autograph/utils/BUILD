load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "type_registry",
    srcs = [
        "type_registry.py",
    ],
    visibility = ["//machina:__subpackages__"],
)

py_strict_library(
    name = "tensor_list",
    srcs = ["tensor_list.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/framework:tensor",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:tensor_array_ops",
    ],
)

py_strict_library(
    name = "ag_logging",
    srcs = ["ag_logging.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/platform:tf_logging",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "context_managers",
    srcs = ["context_managers.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/framework:ops",
        "//machina/python/ops:tensor_array_ops",
    ],
)

py_strict_library(
    name = "misc",
    srcs = ["misc.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
    ],
)

py_strict_library(
    name = "testing",
    srcs = ["testing.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:op_callbacks",
        "//machina/python/framework:ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        ":context_managers",
        ":misc",
        ":tensor_list",
    ],
)

py_strict_library(
    name = "tensors",
    srcs = ["tensors.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:tensor_array_ops",
    ],
)

py_strict_test(
    name = "context_managers_test",
    srcs = ["context_managers_test.py"],
    deps = [
        ":context_managers",
        #internal proto upb dep
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "misc_test",
    srcs = ["misc_test.py"],
    deps = [
        ":misc",
        #internal proto upb dep
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "tensor_list_test",
    srcs = ["tensor_list_test.py"],
    deps = [
        ":tensor_list",
        #internal proto upb dep
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "tensors_test",
    srcs = ["tensors_test.py"],
    deps = [
        ":tensors",
        #internal proto upb dep
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/platform:client_testlib",
    ],
)
