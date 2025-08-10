# QuantizationDebugger for TFLite accuracy tooling.
load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "debugger",
    srcs = ["debugger.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/lite/python:convert",
        "//machina/lite/python:interpreter",
        "//machina/lite/python/metrics",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "debugger_test",
    srcs = [
        "debugger_test.py",
    ],
    deps = [
        ":debugger",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina:machina_py",
        "//machina/lite/python:convert",
        "//machina/lite/python:lite",
        "//machina/lite/python/metrics",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//machina/python/trackable:autotrackable",
    ],
)
