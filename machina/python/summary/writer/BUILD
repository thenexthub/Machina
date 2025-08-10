load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "fake_summary_writer",
    testonly = 1,
    srcs = ["fake_summary_writer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":writer",
        ":writer_cache",
        "//machina/core:protos_all_py",
        "//machina/python/framework:test_lib",
    ],
)

py_strict_library(
    name = "writer",
    srcs = ["writer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":event_file_writer",
        ":event_file_writer_v2",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/summary:plugin_asset",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "event_file_writer",
    srcs = ["event_file_writer.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/client:_pywrap_events_writer",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
    ],
)

py_strict_library(
    name = "writer_cache",
    srcs = ["writer_cache.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":writer",
        "//machina/python/framework:ops",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "event_file_writer_v2",
    srcs = ["event_file_writer_v2.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:summary_ops_v2",
        "//machina/python/platform:gfile",
    ],
)

tf_py_strict_test(
    name = "writer_test",
    size = "small",
    srcs = [
        "writer_test.py",
    ],
    deps = [
        ":writer",
        ":writer_cache",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:summary_ops_v2",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/summary:plugin_asset",
        "//machina/python/summary:summary_iterator",
        "//machina/python/util:compat",
    ],
)
