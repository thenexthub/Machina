# Tests of TensorFlow kernels written using the Python API.

load("//machina:machina.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "bijector_test",
    size = "small",
    srcs = ["bijector_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/distributions:bijector",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "util_test",
    size = "medium",
    srcs = ["util_test.py"],
    shard_count = 3,
    xla_enable_strict_auto_jit = False,  # TODO(b/144920376)
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops/distributions:util",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "kullback_leibler_test",
    size = "small",
    srcs = ["kullback_leibler_test.py"],
    deps = [
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops/distributions:kullback_leibler",
        "//machina/python/ops/distributions:normal",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "beta_test",
    size = "small",
    srcs = ["beta_test.py"],
    tags = [
        "no_oss",  # b/205574093, and because tf.distributions is deprecated
        "nomac",  # b/191763315
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops/distributions:beta",
        "//machina/python/ops/distributions:kullback_leibler",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "bernoulli_test",
    size = "medium",
    srcs = ["bernoulli_test.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops/distributions:bernoulli",
        "//machina/python/ops/distributions:kullback_leibler",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "categorical_test",
    size = "small",
    srcs = ["categorical_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops/distributions:categorical",
        "//machina/python/ops/distributions:kullback_leibler",
        "//machina/python/ops/distributions:normal",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "dirichlet_test",
    size = "small",
    srcs = ["dirichlet_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/distributions:dirichlet",
        "//machina/python/ops/distributions:kullback_leibler",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "dirichlet_multinomial_test",
    size = "medium",
    srcs = ["dirichlet_multinomial_test.py"],
    tags = [
        "noguitar",  # b/110489471
        "notap",  # b/110489471
    ],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/distributions:dirichlet_multinomial",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "exponential_test",
    srcs = ["exponential_test.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops/distributions:exponential",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "gamma_test",
    srcs = ["gamma_test.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops/distributions:gamma",
        "//machina/python/ops/distributions:kullback_leibler",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "laplace_test",
    srcs = ["laplace_test.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops/distributions:laplace",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "multinomial_test",
    srcs = ["multinomial_test.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/distributions:multinomial",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "student_t_test",
    size = "small",
    srcs = ["student_t_test.py"],
    tags = [
        # TODO(b/121223043): Re-enable this test after fixing "mean not defined"
        # errors.
        "no_mac",
        "no_oss",
    ],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops/distributions:student_t",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "uniform_test",
    size = "small",
    srcs = ["uniform_test.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/distributions:uniform",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "normal_test",
    size = "medium",
    srcs = ["normal_test.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/distributions:kullback_leibler",
        "//machina/python/ops/distributions:normal",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "special_math_test",
    size = "medium",
    srcs = ["special_math_test.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops/distributions:special_math",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "identity_bijector_test",
    size = "small",
    srcs = ["identity_bijector_test.py"],
    deps = [
        "//machina/python/framework:test_lib",
        "//machina/python/ops/distributions:bijector_test_util",
        "//machina/python/ops/distributions:identity_bijector",
        "//machina/python/platform:client_testlib",
    ],
)
