load("//machina:strict.default.bzl", "py_strict_binary", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "micro_benchmarks",
    srcs = ["micro_benchmarks.py"],
    deps = [
        ":numpy_mlp",
        ":tf_numpy_mlp",
        "//machina:machina_py_no_contrib",
        "//machina/python:extra_py_tests_deps",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

py_strict_library(
    name = "numpy_mlp",
    srcs = ["numpy_mlp.py"],
    deps = [
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "tf_numpy_mlp",
    srcs = ["tf_numpy_mlp.py"],
    deps = ["//machina:machina_py_no_contrib"],
)
