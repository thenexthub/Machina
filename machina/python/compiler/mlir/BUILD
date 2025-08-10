load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_library(
    name = "mlir",
    srcs = ["mlir.py"],
    deps = [
        "//machina/python:pywrap_mlir",
        "//machina/python/util:tf_export",
    ],
)

py_strict_test(
    name = "mlir_test",
    srcs = ["mlir_test.py"],
    data = [
        "multi_add.tflite",
    ],
    tags = [
        "no_mac",  # TODO(b/291995580): Reenable.
        "no_pip",
    ],
    deps = [
        ":mlir",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/python:pywrap_mlir",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:logging_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
    ],
)
