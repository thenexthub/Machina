# Tests of TensorFlow kernels written using the Python API.

load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "summary_ops_test",
    size = "medium",
    srcs = ["summary_ops_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:tf_record",
        "//machina/python/module",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:summary_ops_v2",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:save",
        "//machina/python/saved_model:tag_constants",
    ],
)

cuda_py_strict_test(
    name = "summary_v1_audio_op_test",
    size = "small",
    srcs = ["summary_v1_audio_op_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/platform:client_testlib",
        "//machina/python/summary:summary_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "summary_v1_image_op_test",
    size = "small",
    srcs = ["summary_v1_image_op_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/platform:client_testlib",
        "//machina/python/summary:summary_py",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "summary_v1_ops_test",
    size = "small",
    srcs = ["summary_v1_ops_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:logging_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/summary:summary_py",
    ],
)

tf_py_strict_test(
    name = "summary_v1_tensor_op_test",
    size = "small",
    srcs = ["summary_v1_tensor_op_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/summary:summary_py",
        "//third_party/py/numpy",
    ],
)
