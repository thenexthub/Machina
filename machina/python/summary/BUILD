load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "plugin_asset",
    srcs = ["plugin_asset.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/framework:ops",
    ],
)

py_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
    visibility = ["//visibility:public"],
)

py_strict_library(
    name = "summary_iterator",
    srcs = ["summary_iterator.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/lib/io:tf_record",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "summary_py",
    srcs = ["summary.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":tb_summary",
        "//machina/core:protos_all_py",
        "//machina/python/distribute:summary_op_util",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:logging_ops_gen",
        "//machina/python/ops:summary_op_util",
        "//machina/python/ops:summary_ops_gen",
        "//machina/python/ops:summary_ops_v2",
        "//machina/python/summary/writer",
        "//machina/python/summary/writer:writer_cache",
        "//machina/python/training:training_util",
        "//machina/python/util:compat",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "plugin_asset_test",
    size = "small",
    srcs = ["plugin_asset_test.py"],
    deps = [
        ":plugin_asset",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "summary_iterator_test",
    size = "small",
    srcs = ["summary_iterator_test.py"],
    deps = [
        ":summary_iterator",
        ":summary_py",
        "//machina/core:protos_all_py",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//machina/python/summary/writer",
    ],
)

tf_py_strict_test(
    name = "summary_test",
    size = "small",
    srcs = ["summary_test.py"],
    deps = [
        ":summary_py",
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:summary_ops_v2",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "summary_v2_test",
    size = "small",
    srcs = ["summary_v2_test.py"],
    deps = [
        ":summary_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:summary_ops_v2",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:training_util",
        "@pypi_tb_nightly//:pkg",
    ],
)

py_strict_library(
    name = "tb_summary",
    srcs = ["tb_summary.py"],
    visibility = ["//machina:internal"],
    deps = ["//machina/python/util:tf_export"],
)
