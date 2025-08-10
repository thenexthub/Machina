# Description:
#   Python library for splitting and joining large protos.
load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "saved_model",
    srcs = ["saved_model.py"],
    # NOTE(yibaimeng): To be removed when everything is migrated to `pywrap_saved_model.Save`.
    visibility = [
        "//machina:internal",
        "//waymo/ml/deploy/machina:__pkg__",
    ],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/tools/proto_splitter:constants",
        "//machina/tools/proto_splitter:split",
        "//machina/tools/proto_splitter:split_graph_def",
    ],
)

py_strict_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.py"],
    tags = [
        "no_mac",  # b/291933687
        "no_windows",  # b/291001524
    ],
    deps = [
        ":saved_model",
        ":test_util",
        #internal proto upb dep
        "//machina/core:protos_all_py",
        "//machina/python/platform:client_testlib",
        "//machina/tools/proto_splitter:constants",
    ],
)

pytype_strict_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_util",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "test_util_test",
    srcs = ["test_util_test.py"],
    deps = [
        #internal proto upb dep
        "//machina/python/framework:dtypes",
        "//machina/python/platform:client_testlib",
        "//machina/tools/proto_splitter/python:test_util",
        "@absl_py//absl/testing:parameterized",
    ],
)
