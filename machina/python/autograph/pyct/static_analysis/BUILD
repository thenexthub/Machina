load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "liveness",
    srcs = ["liveness.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":annos",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:cfg",
        "//machina/python/autograph/pyct:transformer",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "reaching_fndefs",
    srcs = ["reaching_fndefs.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:cfg",
        "//machina/python/autograph/pyct:transformer",
        "@pypi_gast//:pkg",
    ],
)

py_strict_test(
    name = "reaching_fndefs_test",
    srcs = ["reaching_fndefs_test.py"],
    deps = [
        ":activity",
        ":reaching_definitions",
        ":reaching_fndefs",
        #internal proto upb dep
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:cfg",
        "//machina/python/autograph/pyct:naming",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:transformer",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "activity",
    srcs = ["activity.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":annos",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:transformer",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "type_inference",
    srcs = ["type_inference.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":activity",
        ":annos",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:cfg",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:transformer",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "reaching_definitions",
    srcs = ["reaching_definitions.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:cfg",
        "//machina/python/autograph/pyct:transformer",
        "@pypi_gast//:pkg",
    ],
)

py_strict_library(
    name = "annos",
    srcs = ["annos.py"],
    visibility = ["//visibility:public"],
)

py_strict_test(
    name = "activity_test",
    srcs = ["activity_test.py"],
    deps = [
        ":activity",
        ":annos",
        "@pypi_gast//:pkg",
        #internal proto upb dep
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:naming",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:transformer",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "liveness_test",
    testonly = True,
    srcs = ["liveness_test.py"],
    deps = [
        ":activity",
        ":liveness",
        ":reaching_fndefs",
        #internal proto upb dep
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:cfg",
        "//machina/python/autograph/pyct:naming",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:transformer",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "reaching_definitions_test",
    srcs = ["reaching_definitions_test.py"],
    deps = [
        ":activity",
        ":reaching_definitions",
        #internal proto upb dep
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:cfg",
        "//machina/python/autograph/pyct:naming",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:transformer",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "type_inference_test",
    srcs = ["type_inference_test.py"],
    deps = [
        ":activity",
        ":reaching_definitions",
        ":reaching_fndefs",
        ":type_inference",
        #internal proto upb dep
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:cfg",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:transpiler",
        "//machina/python/platform:client_testlib",
    ],
)
