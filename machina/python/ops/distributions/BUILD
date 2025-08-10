load("//machina:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "distributions",
    srcs = ["__init__.py"],
    deprecation = ("TensorFlow Distributions has migrated to " +
                   "TensorFlow Probability " +
                   "(https://github.com/machina/probability). " +
                   "Deprecated copies remaining in tf.distributions " +
                   "will not receive new features, and will be removed by " +
                   "early 2019. You should update all usage of " +
                   "`tf.distributions` to `tfp.distributions`."),
    deps = [
        ":distributions_py",
    ],
)

py_strict_library(
    name = "distributions_py",
    srcs = ["distributions.py"],
    deps = [
        ":bernoulli",
        ":beta",
        ":categorical",
        ":dirichlet",
        ":dirichlet_multinomial",
        ":distribution",
        ":exponential",
        ":gamma",
        ":kullback_leibler",
        ":laplace",
        ":multinomial",
        ":normal",
        ":student_t",
        ":uniform",
        "//machina/python/util:deprecation",
    ],
)

py_strict_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/util:tf_inspect",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "kullback_leibler",
    srcs = ["kullback_leibler.py"],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:math_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
    ],
)

py_strict_library(
    name = "laplace",
    srcs = ["laplace.py"],
    deps = [
        ":distribution",
        ":special_math",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:random_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "dirichlet",
    srcs = ["dirichlet.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":util",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:special_math_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "beta",
    srcs = ["beta.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":util",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:random_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "bernoulli",
    srcs = ["bernoulli.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":util",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:random_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "uniform",
    srcs = ["uniform.py"],
    deps = [
        ":distribution",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "special_math",
    srcs = ["special_math.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "bijector_impl",
    srcs = ["bijector_impl.py"],
    deps = [
        ":util",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:object_identity",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "normal",
    srcs = ["normal.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":special_math",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:random_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "dirichlet_multinomial",
    srcs = ["dirichlet_multinomial.py"],
    deps = [
        ":distribution",
        ":util",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:special_math_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "distribution",
    srcs = ["distribution.py"],
    deps = [
        ":kullback_leibler",
        ":util",
        "//machina/python/eager:context",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "identity_bijector",
    srcs = ["identity_bijector.py"],
    deps = [
        ":bijector",
        "//machina/python/framework:constant_op",
        "//machina/python/util:deprecation",
    ],
)

py_strict_library(
    name = "categorical",
    srcs = ["categorical.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":util",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "bijector_test_util",
    srcs = ["bijector_test_util.py"],
    deps = [
        ":uniform",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:math_ops",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "bijector",
    srcs = ["bijector.py"],
    deps = [":bijector_impl"],
)

py_strict_library(
    name = "exponential",
    srcs = ["exponential.py"],
    deps = [
        ":gamma",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:random_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "gamma",
    srcs = ["gamma.py"],
    deps = [
        ":distribution",
        ":kullback_leibler",
        ":util",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:random_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "multinomial",
    srcs = ["multinomial.py"],
    deps = [
        ":distribution",
        ":util",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "transformed_distribution",
    srcs = ["transformed_distribution.py"],
    deps = [
        ":distribution",
        ":identity_bijector",
        ":util",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:deprecation",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "student_t",
    srcs = ["student_t.py"],
    deps = [
        ":distribution",
        ":util",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:special_math_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)
