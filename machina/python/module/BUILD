load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "module",
    srcs = ["module.py"],
    deps = [
        "//machina/python:tf2",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:ops",
        "//machina/python/ops:variables",
        "//machina/python/trackable:autotrackable",
        "//machina/python/util:nest",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "module_test",
    srcs = ["module_test.py"],
    deps = [
        ":module",
        "//machina/python:extra_py_tests_deps",
        "//machina/python:tf2",
        "//machina/python/distribute:ps_values",
        "//machina/python/distribute:tpu_values",
        "//machina/python/distribute:values",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:type_spec",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)
