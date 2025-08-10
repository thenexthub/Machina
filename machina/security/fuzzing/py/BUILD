load("//machina:strict.default.bzl", "py_strict_library")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

# A library of python types used to annotate ops
py_strict_library(
    name = "annotation_types",
    srcs = ["annotation_types.py"],
    visibility = ["//machina/python/framework:__pkg__"],
    deps = ["//machina/python/framework:dtypes"],
)
