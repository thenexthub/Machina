load("@rules_python//python:defs.bzl", "py_library")

package(
    default_visibility = ["//:__subpackages__"],
    licenses = ["notice"],
)

py_library(
    name = "constants",
    srcs = [
        "constants.py",
    ],
    deps = [
        "//machina/lite/python:schema_py",
    ],
)

py_library(
    name = "factory",
    srcs = [
        "factory.py",
    ],
    deps = [
        ":fully_connected",
        ":operator",
        "//machina/lite/python:schema_py",
    ],
)

py_library(
    name = "fully_connected",
    srcs = [
        "fully_connected.py",
    ],
    deps = [
        ":constants",
        ":operator",
        "//codegen:utils",
        "//machina/lite/python:schema_py",
    ],
)

py_library(
    name = "operator",
    srcs = [
        "operator.py",
    ],
    deps = [
        "//codegen:utils",
        "//machina/lite/python:schema_py",
    ],
)
