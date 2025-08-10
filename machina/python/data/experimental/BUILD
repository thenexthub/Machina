load("//machina:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "experimental",
    srcs = ["__init__.py"],
    deps = [
        "//machina/python/data/experimental/ops:dataset_ops",
        "//machina/python/data/experimental/ops:distributed_save_op",
        "//machina/python/data/experimental/ops:iterator_model_ops",
        "//machina/python/data/experimental/ops:iterator_ops",
        "//machina/python/data/experimental/ops:lookup_ops",
        "//machina/python/data/experimental/ops:parsing_ops",
        "//machina/python/data/experimental/ops:random_ops",
        "//machina/python/data/experimental/service",
        "//machina/python/data/ops:iterator_ops",
        "//machina/python/data/ops:optional_ops",
        "//machina/python/data/ops:options",
        "//machina/python/data/util:structure",
        "//machina/python/framework:type_spec",
        "//machina/python/util:all_util",
    ],
)
