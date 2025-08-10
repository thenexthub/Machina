load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "test_lib",
    srcs = [
        "converter_testing.py",
    ],
    visibility = ["//machina:__subpackages__"],
    deps = [
        ":config",
        ":converter",
        "//machina/python/autograph/impl:api",
        "//machina/python/framework:ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "ag_ctx",
    srcs = ["ag_ctx.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/utils:ag_logging",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "unsupported_features_checker",
    srcs = ["unsupported_features_checker.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/pyct:errors",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "config",
    srcs = ["config.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [":config_lib"],
)

py_strict_library(
    name = "config_lib",
    srcs = ["config_lib.py"],
    visibility = ["//visibility:private"],
)

py_strict_library(
    name = "function_wrappers",
    srcs = ["function_wrappers.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        ":ag_ctx",
        ":converter",
        "//machina/python/autograph/operators:variables",
        "//machina/python/framework:auto_control_deps",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/util:nest",
    ],
)

py_strict_library(
    name = "converter",
    srcs = ["converter.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:ast_util",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:templates",
        "//machina/python/autograph/pyct:transformer",
        "//machina/python/util:tf_export",
    ],
)

py_strict_test(
    name = "converter_test",
    srcs = ["converter_test.py"],
    deps = [
        ":converter",
        ":test_lib",
        #internal proto upb dep
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:loader",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:templates",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "function_wrappers_test",
    srcs = ["function_wrappers_test.py"],
    deps = [
        ":converter",
        ":function_wrappers",
        #internal proto upb dep
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)
