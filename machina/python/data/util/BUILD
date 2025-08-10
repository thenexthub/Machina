load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "nest",
    srcs = ["nest.py"],
    deps = [
        "//machina/python/util:nest_util",
    ],
)

py_strict_test(
    name = "nest_test",
    size = "small",
    srcs = ["nest_test.py"],
    deps = [
        ":nest",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "sparse",
    srcs = ["sparse.py"],
    deps = [
        ":nest",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:sparse_ops",
    ],
)

py_strict_test(
    name = "sparse_test",
    size = "small",
    srcs = ["sparse_test.py"],
    deps = [
        ":nest",
        ":sparse",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "structure",
    srcs = ["structure.py"],
    deps = [
        ":nest",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:none_tensor",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:type_spec",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:tf_logging",
        "//machina/python/types:internal",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:nest_util",
        "//machina/python/util:tf_export",
        "@pypi_wrapt//:pkg",
    ],
)

py_strict_test(
    name = "structure_test",
    size = "small",
    srcs = ["structure_test.py"],
    deps = [
        ":nest",
        ":structure",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/ops/ragged:ragged_tensor_value",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@pypi_wrapt//:pkg",
    ],
)

py_strict_library(
    name = "options",
    srcs = ["options.py"],
    deps = ["@absl_py//absl/logging"],
)

py_strict_test(
    name = "options_test",
    size = "small",
    srcs = ["options_test.py"],
    deps = [
        ":options",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/framework:combinations",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "convert",
    srcs = ["convert.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
    ],
)

py_strict_test(
    name = "convert_test",
    size = "small",
    srcs = ["convert_test.py"],
    deps = [
        ":convert",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "random_seed",
    srcs = ["random_seed.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:random_seed",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
    ],
)

py_strict_test(
    name = "random_seed_test",
    size = "medium",
    srcs = ["random_seed_test.py"],
    deps = [
        ":random_seed",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "traverse",
    srcs = ["traverse.py"],
    deps = [
        "//machina/python/framework:dtypes",
    ],
)

py_strict_test(
    name = "traverse_test",
    size = "small",
    srcs = ["traverse_test.py"],
    deps = [
        ":traverse",
        "//machina/python/compat",
        "//machina/python/data/experimental/ops:dataset_ops",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:combinations",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)
