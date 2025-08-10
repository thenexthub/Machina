load("//machina:strict.default.bzl", "py_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "stack_trace_example",
    srcs = ["stack_trace_example.py"],
    deps = [
        "//machina:machina_py",
        "//machina/lite/python:lite",
        "@absl_py//absl:app",
    ],
)
