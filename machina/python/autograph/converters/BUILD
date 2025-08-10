load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "slices",
    srcs = ["slices.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/lang:directives",
        "//machina/python/autograph/pyct:templates",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "continue_statements",
    srcs = ["continue_statements.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:templates",
        "//machina/python/autograph/pyct/static_analysis:activity",
        "//machina/python/autograph/pyct/static_analysis:annos",
    ],
)

py_strict_library(
    name = "logical_expressions",
    srcs = ["logical_expressions.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:templates",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "break_statements",
    srcs = ["break_statements.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:templates",
        "//machina/python/autograph/pyct/static_analysis:activity",
        "//machina/python/autograph/pyct/static_analysis:annos",
    ],
)

py_strict_library(
    name = "conditional_expressions",
    srcs = ["conditional_expressions.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:templates",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "call_trees",
    srcs = ["call_trees.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:templates",
        "//machina/python/autograph/utils:ag_logging",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "directives",
    srcs = ["directives.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/lang:directives",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/util:tf_inspect",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "variables",
    srcs = ["variables.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:templates",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "control_flow",
    srcs = ["control_flow.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/lang:directives",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:cfg",
        "//machina/python/autograph/pyct:origin_info",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:templates",
        "//machina/python/autograph/pyct/static_analysis:activity",
        "//machina/python/autograph/pyct/static_analysis:annos",
        "//machina/python/autograph/pyct/static_analysis:liveness",
        "//machina/python/autograph/pyct/static_analysis:reaching_definitions",
        "//machina/python/autograph/pyct/static_analysis:reaching_fndefs",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "functions",
    srcs = ["functions.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:templates",
        "//machina/python/autograph/pyct/static_analysis:activity",
        "//machina/python/autograph/pyct/static_analysis:annos",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "lists",
    srcs = ["lists.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/lang:directives",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:templates",
        "//machina/python/autograph/pyct/static_analysis:activity",
        "//machina/python/autograph/pyct/static_analysis:annos",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "return_statements",
    srcs = ["return_statements.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:templates",
        "//machina/python/autograph/pyct/static_analysis:activity",
        "//machina/python/autograph/pyct/static_analysis:annos",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "asserts",
    srcs = ["asserts.py"],
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/pyct:templates",
        "@pypi_gast//:pkg",
    ],
)

py_strict_test(
    name = "asserts_test",
    srcs = ["asserts_test.py"],
    deps = [
        ":asserts",
        ":functions",
        ":return_statements",
        #internal proto upb dep
        "//machina/python/autograph/core:test_lib",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "break_statements_test",
    srcs = ["break_statements_test.py"],
    deps = [
        ":break_statements",
        #internal proto upb dep
        "//machina/python/autograph/core:test_lib",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "call_trees_test",
    srcs = ["call_trees_test.py"],
    deps = [
        ":call_trees",
        ":functions",
        #internal proto upb dep
        "//machina/python/autograph/core:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "conditional_expressions_test",
    srcs = ["conditional_expressions_test.py"],
    deps = [
        ":conditional_expressions",
        #internal proto upb dep
        "//machina/python/autograph/core:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "continue_statements_test",
    srcs = ["continue_statements_test.py"],
    deps = [
        ":continue_statements",
        #internal proto upb dep
        "//machina/python/autograph/core:test_lib",
        "//machina/python/framework:ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "control_flow_test",
    srcs = ["control_flow_test.py"],
    deps = [
        ":break_statements",
        ":continue_statements",
        ":control_flow",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina/python/autograph/core:test_lib",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_util",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:nest",
    ],
)

py_strict_test(
    name = "directives_test",
    srcs = ["directives_test.py"],
    deps = [
        ":directives",
        #internal proto upb dep
        "//machina/python/autograph/core:test_lib",
        "//machina/python/autograph/lang:directives",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "functions_test",
    srcs = ["functions_test.py"],
    deps = [
        ":functions",
        ":return_statements",
        #internal proto upb dep
        "//machina/python/autograph/core:ag_ctx",
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/core:test_lib",
        "//machina/python/autograph/impl:api",
        "//machina/python/framework:constant_op",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "lists_test",
    srcs = ["lists_test.py"],
    deps = [
        ":directives",
        ":lists",
        #internal proto upb dep
        "//machina/python/autograph/core:test_lib",
        "//machina/python/autograph/lang:directives",
        "//machina/python/autograph/lang:special_functions",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:list_ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "logical_expressions_test",
    srcs = ["logical_expressions_test.py"],
    deps = [
        ":logical_expressions",
        #internal proto upb dep
        "//machina/python/autograph/core:test_lib",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "return_statements_test",
    srcs = ["return_statements_test.py"],
    deps = [
        ":functions",
        ":return_statements",
        #internal proto upb dep
        "//machina/python/autograph/core:test_lib",
        "//machina/python/framework:ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "slices_test",
    srcs = ["slices_test.py"],
    deps = [
        ":directives",
        ":slices",
        #internal proto upb dep
        "//machina/python/autograph/core:test_lib",
        "//machina/python/autograph/lang:directives",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:list_ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "variables_test",
    srcs = ["variables_test.py"],
    deps = [
        ":variables",
        #internal proto upb dep
        "//machina/python/autograph/core:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)
