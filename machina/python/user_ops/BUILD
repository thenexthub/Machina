# Description:
#   Contains User Ops (internal TensorFlow version).

load("//machina:strict.default.bzl", "py_strict_library")
load("//machina/python:build_defs.bzl", "tf_gen_op_wrapper_private_py")

visibility = [
    "//machina:internal",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

tf_gen_op_wrapper_private_py(
    name = "user_ops_gen",
    out = "ops/gen_user_ops.py",
    py_lib_rule = py_strict_library,
)

# This target is deprecated.
py_strict_library(
    name = "ops",
    srcs = ["user_ops.py"],
    deps = [
        ":user_ops_gen",
        "//machina/python/util:tf_export",
    ],
)
