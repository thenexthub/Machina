load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//machina/python/tpu:tpu.bzl", "tpu_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_library(
    name = "xla_sharding",
    srcs = ["xla_sharding.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/tf2xla/python:xla",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/ops:resource_variable_ops",
        "//third_party/py/numpy",
        "@local_xla//xla:xla_data_proto_py",
    ],
)

py_strict_test(
    name = "xla_sharding_test",
    srcs = ["xla_sharding_test.py"],
    deps = [
        ":xla_sharding",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "@local_xla//xla:xla_data_proto_py",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:variables",
        "@absl_py//absl/testing:absltest",
    ],
)

tpu_py_strict_test(
    name = "resource_variable_xla_sharding_test",
    srcs = ["resource_variable_xla_sharding_test.py"],
    disable_v3_4chips = False,
    tags = ["requires-net:external"],
    deps = [
        ":xla_sharding",
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/tpu:device_assignment",
        "//machina/python/tpu:tpu_py",
        "//machina/python/training:adagrad",
    ],
)
