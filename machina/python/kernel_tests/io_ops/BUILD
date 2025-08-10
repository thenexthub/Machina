# Tests of TensorFlow IO kernels written using the Python API.

load("//machina:machina.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_py_strict_test(
    name = "checkpoint_ops_test",
    size = "medium",
    srcs = ["checkpoint_ops_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:checkpoint_ops_gen",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:flags",
        "//machina/python/training:saver",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "decode_csv_op_test",
    size = "small",
    srcs = ["decode_csv_op_test.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/ops:parsing_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "io_ops_test",
    size = "small",
    srcs = ["io_ops_test.py"],
    deps = [
        "//machina/python/framework:test_lib",
        "//machina/python/ops:io_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "parse_single_example_op_test",
    size = "small",
    srcs = ["parse_single_example_op_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:parsing_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "parsing_ops_test",
    size = "medium",
    srcs = ["parsing_ops_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops/ragged:ragged_concat_ops",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "reader_ops_test",
    size = "small",
    srcs = ["reader_ops_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:tf_record",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "record_input_test",
    size = "medium",
    srcs = ["record_input_test.py"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:tf_record",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "save_restore_ops_test",
    size = "small",
    srcs = ["save_restore_ops_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:io_ops_gen",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)
