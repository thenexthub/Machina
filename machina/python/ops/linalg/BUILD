load("//machina:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "linalg_impl",
    srcs = ["linalg_impl.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:linalg_ops_gen",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:special_math_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_util",
    srcs = ["linear_operator_util.py"],
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_low_rank_update",
    srcs = ["linear_operator_low_rank_update.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_diag",
        ":linear_operator_identity",
        ":linear_operator_util",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator_circulant",
    srcs = ["linear_operator_circulant.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        ":property_hint_util",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/distributions:util",
        "//machina/python/ops/signal:fft_ops",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_identity",
    srcs = ["linear_operator_identity.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_diag",
        ":linear_operator_util",
        ":property_hint_util",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_kronecker",
    srcs = ["linear_operator_kronecker.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        "//machina/python/framework:common_shapes",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "property_hint_util",
    srcs = ["property_hint_util.py"],
)

py_strict_library(
    name = "linear_operator_block_lower_triangular",
    srcs = ["linear_operator_block_lower_triangular.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_addition",
        ":linear_operator_full_matrix",
        ":linear_operator_identity",
        ":linear_operator_util",
        "//machina/python/framework:common_shapes",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator",
    srcs = [
        "linear_operator.py",
        "linear_operator_adjoint.py",
        "linear_operator_composition.py",
        "linear_operator_inversion.py",
        "linear_operator_lower_triangular.py",
    ],
    deps = [
        ":linalg_impl",
        ":linear_operator_util",
        ":property_hint_util",
        ":slicing",
        "//machina/python/framework:common_shapes",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:composite_tensor_gradient",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:type_spec",
        "//machina/python/framework:type_spec_registry",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
        "//machina/python/trackable:data_structures",
        "//machina/python/util:deprecation",
        "//machina/python/util:dispatch",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "//machina/python/util:variable_utils",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_tridiag",
    srcs = ["linear_operator_tridiag.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:manip_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator_toeplitz",
    srcs = ["linear_operator_toeplitz.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_circulant",
        ":linear_operator_util",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/signal:fft_ops",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator_block_diag",
    srcs = ["linear_operator_block_diag.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_util",
        ":property_hint_util",
        "//machina/python/framework:common_shapes",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator_test_util",
    srcs = ["linear_operator_test_util.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator_util",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:test_lib",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:sort_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_v2",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:nested_structure_coder",
        "//machina/python/saved_model:save",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_permutation",
    srcs = ["linear_operator_permutation.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:sort_ops",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linalg",
    srcs = ["linalg.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_block_diag",
        ":linear_operator_block_lower_triangular",
        ":linear_operator_circulant",
        ":linear_operator_diag",
        ":linear_operator_full_matrix",
        ":linear_operator_householder",
        ":linear_operator_identity",
        ":linear_operator_kronecker",
        ":linear_operator_low_rank_update",
        ":linear_operator_permutation",
        ":linear_operator_toeplitz",
        ":linear_operator_tridiag",
        ":linear_operator_zeros",
    ],
)

py_strict_library(
    name = "linear_operator_full_matrix",
    srcs = ["linear_operator_full_matrix.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_util",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator_householder",
    srcs = ["linear_operator_householder.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "slicing",
    srcs = ["slicing.py"],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_zeros",
    srcs = ["linear_operator_zeros.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_diag",
    srcs = ["linear_operator_diag.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        ":property_hint_util",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator_addition",
    srcs = ["linear_operator_addition.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_diag",
        ":linear_operator_full_matrix",
        ":linear_operator_identity",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
    ],
)
