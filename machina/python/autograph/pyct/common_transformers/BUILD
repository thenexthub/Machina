load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "common_transformers",
    srcs = [
        "anf.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/autograph/pyct:gast_util",
        "//machina/python/autograph/pyct:templates",
        "//machina/python/autograph/pyct:transformer",
        "@pypi_gast//:pkg",
    ],
)

py_strict_test(
    name = "anf_test",
    srcs = ["anf_test.py"],
    tags = ["no_oss"],
    deps = [
        ":common_transformers",
        "@pypi_gast//:pkg",
        #internal proto upb dep
        "//machina/python/autograph/pyct:loader",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:transformer",
        "//machina/python/platform:client_testlib",
    ],
)
