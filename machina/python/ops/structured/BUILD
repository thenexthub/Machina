# Structured Tensors

load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
        "//machina:structured_tensor_allowlist",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "structured",
    srcs = ["__init__.py"],
    tags = ["nofixdeps"],
    deps = [
        ":structured_ops",
    ],
)

py_strict_library(
    name = "structured_ops",
    srcs = ["structured_ops.py"],
    deps = [
        ":structured_array_ops",
        ":structured_tensor",
    ],
)

py_strict_library(
    name = "structured_tensor",
    srcs = ["structured_tensor.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:extension_type",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:type_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/ragged:dynamic_ragged_shape",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/ops/ragged:row_partition",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "structured_tensor_dynamic",
    srcs = [
        "structured_tensor_dynamic.py",
    ],
    deps = [
        ":structured_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops/ragged:dynamic_ragged_shape",
    ],
)

py_strict_library(
    name = "structured_array_ops",
    srcs = [
        "structured_array_ops.py",
    ],
    deps = [
        ":structured_tensor",
        "//machina/core/config:flags_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops/ragged:dynamic_ragged_shape",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/ops/ragged:row_partition",
        "//machina/python/util:deprecation",
        "//machina/python/util:dispatch",
    ],
)

py_strict_test(
    name = "structured_tensor_test",
    srcs = ["structured_tensor_test.py"],
    deps = [
        ":structured_array_ops",
        ":structured_tensor",
        ":structured_tensor_dynamic",
        "@absl_py//absl/testing:parameterized",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:extension_type",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops/ragged:dynamic_ragged_shape",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/ops/ragged:row_partition",
        "//machina/python/platform:test",
        "//machina/python/util:dispatch",
    ],
)

py_strict_test(
    name = "structured_tensor_spec_test",
    srcs = ["structured_tensor_spec_test.py"],
    deps = [
        ":structured_tensor",
        "@absl_py//absl/testing:parameterized",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:type_spec",
        "//machina/python/ops/ragged:dynamic_ragged_shape",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/ops/ragged:row_partition",
        "//machina/python/platform:test",
    ],
)

py_strict_test(
    name = "structured_array_ops_test",
    srcs = ["structured_array_ops_test.py"],
    deps = [
        ":structured_array_ops",
        ":structured_tensor",
        "@absl_py//absl/testing:parameterized",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops/ragged:dynamic_ragged_shape",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/ops/ragged:row_partition",
        "//machina/python/platform:test",
        "//machina/python/util:nest",
    ],
)

py_strict_test(
    name = "structured_tensor_slice_test",
    srcs = ["structured_tensor_slice_test.py"],
    deps = [
        ":structured_tensor",
        "@absl_py//absl/testing:parameterized",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
    ],
)
