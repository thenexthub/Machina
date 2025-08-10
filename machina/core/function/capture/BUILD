load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
)

py_strict_library(
    name = "free_vars_detect",
    srcs = [
        "free_vars_detect.py",
    ],
    deps = [
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:inspect_utils",
        "//machina/python/autograph/pyct:naming",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:transformer",
        "//machina/python/autograph/pyct/static_analysis:activity",
    ],
)

py_strict_test(
    name = "free_vars_detect_test",
    srcs = ["free_vars_detect_test.py"],
    tags = [
        "no_oss",  # TODO(b/247102978)
    ],
    deps = [
        ":free_vars_detect",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina/python/util:tf_decorator_py",
    ],
)

py_strict_test(
    name = "by_ref_capture_test",
    srcs = ["by_ref_capture_test.py"],
    deps = [
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//machina/python/compat:v2_compat",
        "//machina/python/eager:def_function",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "capture_container",
    srcs = [
        "capture_container.py",
    ],
    deps = [
        "//machina/core/function/trace_type",
        "//machina/python:pywrap_tfe",
        "//machina/python/framework:dtypes",
        "//machina/python/types:core",
        "//machina/python/util:object_identity",
    ],
)

py_strict_test(
    name = "capture_container_test",
    srcs = ["capture_container_test.py"],
    deps = [
        ":capture_container",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//machina/core/function/trace_type",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "restore_captures",
    srcs = ["restore_captures.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/core/function/polymorphism:function_type",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:handle_data_util",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/trackable:asset",
        "//machina/python/trackable:resource",
    ],
)
