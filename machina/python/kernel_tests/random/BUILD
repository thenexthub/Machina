# Tests of TensorFlow kernels written using the Python API.

load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

# CPU only tests should use tf_py_test, GPU tests use cuda_py_test
# Please avoid the py_tests and cuda_py_tests (plural) while we
# fix the shared/overbroad dependencies.

tf_py_strict_test(
    name = "candidate_sampler_ops_test",
    size = "small",
    srcs = ["candidate_sampler_ops_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:candidate_sampling_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "multinomial_op_big_test",
    size = "medium",
    srcs = ["multinomial_op_big_test.py"],
    shard_count = 3,
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "multinomial_op_test",
    size = "small",
    srcs = ["multinomial_op_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "parameterized_truncated_normal_op_test",
    size = "medium",
    srcs = ["parameterized_truncated_normal_op_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:backprop",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

# TODO(b/130359919): Reenable test when it becomes stable
tf_py_strict_test(
    name = "random_binomial_test",
    size = "medium",
    srcs = ["random_binomial_test.py"],
    shard_count = 3,
    tags = ["no_oss"],
    deps = [
        ":util",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:stateful_random_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "random_crop_test",
    size = "small",
    srcs = ["random_crop_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/framework:test_lib",
        "//machina/python/ops:random_crop_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "random_gamma_test",
    size = "medium",
    srcs = ["random_gamma_test.py"],
    shard_count = 4,
    tags = [
        "no_windows",  # TODO(b/218876472)
        "nozapfhahn",
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
        "nomsan",  # b/216320808
    ],
    deps = [
        ":util",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "random_grad_test",
    size = "small",
    srcs = ["random_grad_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_grad",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:stateless_random_ops_v2_gen",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "random_ops_test",
    size = "medium",
    srcs = ["random_ops_test.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:random_ops_gen",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "random_poisson_test",
    size = "medium",
    srcs = ["random_poisson_test.py"],
    deps = [
        ":util",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "random_shuffle_queue_test",
    size = "medium",
    srcs = ["random_shuffle_queue_test.py"],
    tags = [
        "no_cuda_on_cpu_tap",  # TODO(b/171060960) flakyly broken assertions
    ],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "stateless_random_ops_test",
    size = "medium",
    srcs = ["stateless_random_ops_test.py"],
    shard_count = 10,
    xla_tags = [
        "no_cuda_asan",  # times-out
    ],
    deps = [
        "//machina/python/compat",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops:stateless_random_ops_v2_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "stateful_random_ops_test",
    size = "medium",
    srcs = ["stateful_random_ops_test.py"],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        ":util",
        "//machina/python/checkpoint",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:logging_ops",
        "//machina/python/ops:random_ops_gen",
        "//machina/python/ops:random_ops_util",
        "//machina/python/ops:stateful_random_ops",
        "//machina/python/ops:stateful_random_ops_gen",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "random_index_shuffle_test",
    srcs = ["random_index_shuffle_test.py"],
    shard_count = 10,
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_index_shuffle_ops_gen",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//machina/python/ops/distributions:special_math",
        "//third_party/py/numpy",
    ],
)
