load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

py_strict_library(
    name = "dlpack",
    srcs = ["dlpack.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python:pywrap_tfe",
        "//machina/python/eager:context",
        "//machina/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "dlpack_test",
    srcs = ["dlpack_test.py"],
    deps = [
        ":dlpack",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)
