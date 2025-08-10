load("//machina:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "data",
    srcs = ["__init__.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python/data/experimental",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:iterator_ops",
        "//machina/python/data/ops:options",
        "//machina/python/data/ops:readers",
    ],
)
