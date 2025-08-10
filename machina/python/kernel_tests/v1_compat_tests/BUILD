load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_py_strict_test(
    name = "identity_op_py_test",
    size = "small",
    srcs = ["identity_op_py_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "scatter_nd_ops_test",
    size = "small",
    srcs = ["scatter_nd_ops_test.py"],
    deps = [
        "//machina/python/framework:test_lib",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "session_ops_test",
    size = "small",
    srcs = ["session_ops_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:session_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "array_ops_test",
    size = "small",
    srcs = ["array_ops_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "stack_op_test",
    size = "small",
    srcs = ["stack_op_test.py"],
    deps = [
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "dense_update_ops_test",
    size = "medium",
    srcs = ["dense_update_ops_test.py"],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:client_testlib",
    ],
)
