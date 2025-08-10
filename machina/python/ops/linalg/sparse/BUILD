load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_library")

# Description: Sparse CSR support for TensorFlow.
load("//machina:machina.bzl", "tf_gen_op_wrapper_py")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

tf_gen_op_wrapper_py(
    name = "gen_sparse_csr_matrix_ops",
    out = "gen_sparse_csr_matrix_ops.py",
    api_def_srcs = ["//machina/core/api_def:base_api_def"],
    extra_py_deps = [
        "//machina/python:pywrap_tfe",
        "//machina/python/util:dispatch",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    visibility = ["//visibility:private"],
    deps = ["//machina/core:sparse_csr_matrix_ops_op_lib"],
)

py_strict_library(
    name = "sparse_py",
    srcs = ["sparse.py"],
    deps = [
        ":conjugate_gradient",
        ":sparse_csr_matrix_grad",
        ":sparse_csr_matrix_ops",
    ],
)

py_strict_library(
    name = "sparse_csr_matrix_grad",
    srcs = ["sparse_csr_matrix_grad.py"],
    deps = [
        ":sparse_csr_matrix_ops",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:sparse_ops",
    ],
)

py_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

pytype_strict_library(
    name = "conjugate_gradient",
    srcs = ["conjugate_gradient.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/ops/linalg:linalg_impl",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "sparse_csr_matrix_ops",
    srcs = ["sparse_csr_matrix_ops.py"],
    deps = [
        ":gen_sparse_csr_matrix_ops",
        "//machina/python/eager:context",
        "//machina/python/framework:cpp_shape_inference_proto_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
    ],
)
