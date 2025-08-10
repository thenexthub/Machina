load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "authoring",
    srcs = [
        "authoring.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "//machina/lite/python:convert",
        "//machina/lite/python:lite",
        "//machina/python/util:tf_export",
    ],
)

py_strict_test(
    name = "authoring_test",
    srcs = ["authoring_test.py"],
    deps = [
        ":authoring",
        #internal proto upb dep
        "//machina:machina_py",
        "//machina/lite/python:lite",
    ],
)
