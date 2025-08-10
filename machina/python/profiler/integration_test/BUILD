load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_library(
    name = "mnist_testing_utils",
    srcs = ["mnist_testing_utils.py"],
    deps = [
        "//machina:machina_py",
        "//machina/python:extra_py_tests_deps",
    ],
)

cuda_py_strict_test(
    name = "profiler_api_test",
    srcs = ["profiler_api_test.py"],
    tags = [
        "no_oss",  # TODO(b/283175845)
        "no_pip",
        "no_windows",  # TODO(b/192257727)
    ],
    deps = [
        ":mnist_testing_utils",
        "//machina/python/distribute:collective_all_reduce_strategy",
        "//machina/python/distribute:multi_process_runner",
        "//machina/python/eager:context",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:tf_logging",
        "//machina/python/profiler:profiler_client",
        "//machina/python/profiler:profiler_v2",
        "@pypi_portpicker//:pkg",
    ],
)
