load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.bzl", "tf_gen_op_wrapper_py")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/compiler/tf2xla:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "xla_ops",
    srcs = ["xla_ops.cc"],
    deps = [
        "//machina/core:framework",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "xla_ops_wrapper_py",
    out = "xla_ops.py",
    extra_py_deps = [
        "//machina/python:pywrap_tfe",
        "//machina/python/util:dispatch",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    deps = ["//machina/compiler/jit/ops:xla_ops"],
)

py_strict_library(
    name = "xla_ops_grad",
    srcs = ["xla_ops_grad.py"],
    visibility = [
        "//machina/compiler/tf2xla:internal",
        "//machina/python/ops:__pkg__",
    ],
    deps = ["//machina/python/framework:ops"],
)
