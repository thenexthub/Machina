load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

# Description:
#   Tensorflow builder compatibility checker.

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/tools/machina_builder:__subpackages__",
    ],
)

licenses(["notice"])

py_strict_library(
    name = "compat_checker",
    srcs = ["compat_checker.py"],
    deps = [
        "//machina/python/platform:tf_logging",
        "//machina/python/util:tf_inspect",
    ],
)

py_strict_test(
    name = "compat_checker_test",
    srcs = ["compat_checker_test.py"],
    data = [
        ":test_config",
    ],
    tags = ["no_pip"],
    deps = [
        ":compat_checker",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/python/platform:test",
    ],
)

filegroup(
    name = "test_config",
    srcs = ["test_config.ini"],
)
