load("//machina:pytype.default.bzl", "pytype_strict_library")
load(
    "//machina:machina.default.bzl",
    "tf_py_strict_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["@stablehlo//:license"],
    default_visibility = [
        "//machina:__pkg__",
        "//machina/compiler/mlir/quantization/stablehlo/python:internal_visibility_allowlist_package",
    ],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "testing",
    srcs = ["testing.py"],
    tags = ["no_pip"],
    visibility = ["//visibility:public"],
)

tf_py_strict_test(
    name = "testing_test",
    srcs = ["testing_test.py"],
    deps = [
        ":testing",
        "//machina/python/platform:client_testlib",
    ],
)
