load("//machina:strict.default.bzl", "py_strict_binary", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "mnist_tflite",
    srcs = ["mnist_tflite.py"],
    deps = [
        ":dataset",
        "//machina:machina_py",
        "//machina/lite/python:lite",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        "//machina:machina_py",
        "//third_party/py/numpy",
    ],
)
