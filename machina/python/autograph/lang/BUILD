load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "directives",
    srcs = ["directives.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "special_functions",
    srcs = ["special_functions.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/operators:data_structures",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:tensor_util",
    ],
)

py_strict_test(
    name = "special_functions_test",
    srcs = ["special_functions_test.py"],
    deps = [
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina/python/autograph/lang:special_functions",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:list_ops",
        "//machina/python/platform:client_testlib",
    ],
)
