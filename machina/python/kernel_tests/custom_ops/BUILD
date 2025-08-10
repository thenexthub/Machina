# Tests of TensorFlow custom ops written

load("//machina:machina.bzl", "tf_custom_op_library")
load("//machina:machina.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

# Custom op tests
tf_custom_op_library(
    name = "ackermann_op.so",
    srcs = ["ackermann_op.cc"],
)

tf_py_strict_test(
    name = "ackermann_test",
    size = "small",
    srcs = ["ackermann_test.py"],
    data = [":ackermann_op.so"],
    tags = [
        "no_pip",
        "notap",
    ],
    deps = [
        "//machina/python/framework:load_library",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
    ],
)

tf_custom_op_library(
    name = "duplicate_op.so",
    srcs = ["duplicate_op.cc"],
)

tf_py_strict_test(
    name = "duplicate_op_test",
    size = "small",
    srcs = ["duplicate_op_test.py"],
    data = [":duplicate_op.so"],
    tags = [
        "no_pip",
        "notap",
    ],
    deps = [
        "//machina/python/framework:load_library",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
    ],
)

tf_custom_op_library(
    name = "invalid_op.so",
    srcs = ["invalid_op.cc"],
)

tf_py_strict_test(
    name = "invalid_op_test",
    size = "small",
    srcs = ["invalid_op_test.py"],
    data = [":invalid_op.so"],
    tags = [
        "no_pip",
        "notap",
    ],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/framework:load_library",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
    ],
)
