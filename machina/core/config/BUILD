load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_test")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "filegroup")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

exports_files(
    ["flags_api_wrapper.cc"],
    visibility = ["//machina:internal"],
)

cc_library(
    name = "flags_headers",
    textual_hdrs = [
        "flag_defs.h",
        "flags.h",
    ],
    visibility = ["//machina:internal"],
    deps = ["//machina/core/platform:stringpiece"],
)

filegroup(
    name = "flags_headers_filegroup",
    srcs = [
        "flag_defs.h",
        "flags.h",
    ],
    visibility = ["//machina:internal"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "flag_defs.cc",
        "flag_defs.h",
        "flags.cc",
        "flags.h",
    ],
    visibility = ["//machina:internal"],
)

cc_library(
    name = "flag_defs",
    srcs = ["flag_defs.cc"],
    hdrs = ["flag_defs.h"],
    visibility = ["//machina:internal"],
    deps = [
        ":flags",
    ],
)

cc_library(
    name = "flags",
    srcs = ["flags.cc"],
    hdrs = ["flags.h"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/core/platform:stringpiece",
        "//machina/core/util:env_var",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "flags_test",
    srcs = ["flags_test.cc"],
    deps = [
        ":flag_defs",
        ":flags",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

pytype_strict_library(
    name = "flags_py",
    srcs = ["flags.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python:flags_pybind",
        "//machina/python:pywrap_machina",
    ],
)

py_strict_test(
    name = "flags_py_test",
    size = "small",
    srcs = ["flags_test.py"],
    main = "flags_test.py",
    deps = [
        ":flags_py",
        #internal proto upb dep
        "//machina/python/platform:client_testlib",
    ],
)
