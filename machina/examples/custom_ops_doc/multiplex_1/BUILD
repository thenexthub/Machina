# Build multiplex_1 custom ops examples, which is similar to np.where

load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.bzl", "tf_custom_op_library")
load("//machina:machina.default.bzl", "tf_py_test")

licenses(["notice"])

tf_custom_op_library(
    name = "multiplex_1_kernel.so",
    srcs = [
        "multiplex_1_kernel.cc",
        "multiplex_1_op.cc",
    ],
)

py_strict_library(
    name = "multiplex_1_op",
    srcs = ["multiplex_1_op.py"],
    data = [":multiplex_1_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//machina:machina_py",
    ],
)

tf_py_test(
    name = "multiplex_1_test",
    size = "medium",
    srcs = ["multiplex_1_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
    ],
    deps = [
        ":multiplex_1_op",
        "//machina:machina_py",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)
