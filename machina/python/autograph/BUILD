load("//machina:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "autograph",
    srcs = [
        "__init__.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/impl:api",
        "//machina/python/autograph/lang:directives",
        "//machina/python/autograph/lang:special_functions",
        "//machina/python/autograph/operators:__init__",
        "//machina/python/autograph/pyct/common_transformers",
        "//machina/python/autograph/utils:__init__",
        "//machina/python/autograph/utils:ag_logging",
        "//machina/python/util:all_util",
    ],
)
