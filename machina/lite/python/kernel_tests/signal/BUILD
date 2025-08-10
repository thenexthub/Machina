load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        "//machina/lite/python:interpreter",
        "//machina/lite/python:lite",
        "//machina/python/eager:def_function",
    ],
)

cuda_py_strict_test(
    name = "window_ops_test",
    srcs = ["window_ops_test.py"],
    shard_count = 4,
    tags = [
        "no_rocm",
        "no_windows_gpu",
    ],
    deps = [
        ":test_util",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops/signal:window_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
