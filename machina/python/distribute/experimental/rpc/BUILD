# Python bindings for RPC client and server ops.

load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:machina.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "rpc_ops",
    srcs = [
        "rpc_ops.py",
    ],
    deps = [
        "//machina/distribute/experimental/rpc/kernels:gen_rpc_ops",
        "//machina/distribute/experimental/rpc/proto:tf_rpc_service_proto_py",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:none_tensor",
        "//machina/python/framework:type_spec",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/saved_model:nested_structure_coder",
        "//machina/python/types:core",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "rpc_ops_test",
    size = "medium",
    srcs = ["rpc_ops_test.py"],
    shard_count = 7,
    tags = [
        "no_mac",  # flaky, b/205156709
    ],
    deps = [
        ":rpc_ops",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:nest",
        "@pypi_portpicker//:pkg",
    ],
)
