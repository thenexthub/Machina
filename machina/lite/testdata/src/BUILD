load("//machina:strict.default.bzl", "py_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "intermediate_tensor_output",
    srcs = ["intermediate_tensor_output.py"],
    python_version = "PY3",
    deps = [
        "//machina:machina_py",
        "//machina/lite/python:lite",
    ],
)
