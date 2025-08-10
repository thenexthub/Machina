# Description:
# Comprehensive tests for the TensorFlow SavedModels.

load("//machina:machina.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_py_strict_test(
    name = "variable_wrapper_test",
    srcs = ["variable_wrapper_test.py"],
    deps = [
        "//machina/python/checkpoint",
        "//machina/python/eager:test",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:save",
        "//machina/python/trackable:autotrackable",
        "//machina/python/trackable:base",
    ],
)
