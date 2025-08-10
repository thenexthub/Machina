load("//machina:strict.default.bzl", "py_strict_test")
load("//machina:machina.default.bzl", "cuda_py_strict_test")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

licenses(["notice"])

py_strict_test(
    name = "public_symbol_test",
    srcs = ["public_symbol_test.py"],
    deps = [
        "//machina:machina_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "np_config_test",
    srcs = ["np_config_test.py"],
    deps = [
        "//machina:machina_py_no_contrib",
        "//machina/python/ops/numpy_ops:np_config",
    ],
)
