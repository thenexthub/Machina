load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_cuda_cc_test")
load("//machina/compiler/aot:tfcompile.bzl", "tf_library")
load("//machina/compiler/tests:build_combined_defs.bzl", "tf_xla_combined_py_test")
load("//machina/compiler/tests:build_defs.bzl", "generate_backend_suites", "tf_xla_py_strict_test")
load(
    "//machina/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")
load("//machina/python/tpu:tpu.bzl", "tpu_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    includes = [
        "//machina/compiler/tf2xla:internal",
    ],
)

package_group(
    name = "friends",
    includes = [
        "//machina/compiler/tf2xla:friends",
    ],
    packages = [
        # To pass open source testing in the pip Kokoros.
        "//platforms/xla/tests/neural_nets",
    ],
)

generate_backend_suites()

py_strict_library(
    name = "xla_test",
    testonly = 1,
    srcs = ["xla_test.py"],
    visibility = [":friends"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/compiler/xla:compiler_py",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:flags",
        "//machina/python/platform:tf_logging",
        "//machina/python/tpu:tpu_py",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "test_utils",
    testonly = 1,
    srcs = [
        "__init__.py",
        "test_utils.py",
    ],
    deps = [
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_strict_test(
    name = "xla_test_test",
    size = "small",
    srcs = ["xla_test_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//machina/python/platform:client_testlib",
    ],
)

#LINT.IfChange(combined_tests)
# If you add a new tf_xla_py_strict_test please either add the test file to one of the combined test
# targets that matches in all tags and other settings or add a new combined test target.

tf_xla_combined_py_test(
    name = "combined_ops_test_a",
    size = "medium",
    package = "machina.compiler.tests",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    tests = [
        # go/keep-sorted start
        ":adadelta_test_lib",
        ":adagrad_test_lib",
        ":bincount_op_test_lib",
        ":bucketize_op_test_lib",
        ":clustering_test_lib",
        ":const_arg_test_lib",
        ":conv2d_test_lib",
        ":conv3d_test_lib",
        ":data_format_ops_test_lib",
        ":dynamic_slice_ops_test_lib",
        ":dynamic_stitch_test_lib",
        # go/keep-sorted end
    ],
)

tf_xla_combined_py_test(
    name = "combined_ops_test_b",
    size = "medium",
    timeout = "long",
    package = "machina.compiler.tests",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    tests = [
        # go/keep-sorted start
        ":extract_image_patches_op_test_lib",
        ":fake_quant_ops_test_lib",
        ":fifo_queue_test_lib",
        ":ftrl_test_lib",
        ":function_test_lib",
        ":fused_batchnorm_test_lib",
        ":gather_nd_op_test_lib",
        ":lrn_ops_test_lib",
        ":manip_ops_test_lib",
        # go/keep-sorted end
    ],
)

tf_xla_combined_py_test(
    name = "combined_ops_test_c",
    size = "medium",
    timeout = "long",
    package = "machina.compiler.tests",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    tests = [
        # go/keep-sorted start
        ":matrix_inverse_op_test_lib",
        ":matrix_solve_op_test_lib",
        ":momentum_test_lib",
        ":nary_ops_test_lib",
        ":nullary_ops_test_lib",
        ":placeholder_test_lib",
        ":pooling_ops_3d_test_lib",
        ":pooling_ops_test_lib",
        ":proximal_adagrad_test_lib",
        ":proximal_gradient_descent_test_lib",
        # go/keep-sorted end
    ],
)

tf_xla_combined_py_test(
    name = "combined_ops_test_d",
    size = "medium",
    timeout = "long",
    package = "machina.compiler.tests",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    tests = [
        # go/keep-sorted start
        ":quantized_ops_test_lib",
        ":random_ops_test_lib",
        ":reduce_ops_test_lib",
        ":reduce_window_test_lib",
        ":reverse_ops_test_lib",
        ":rmsprop_test_lib",
        ":searchsorted_op_test_lib",
        ":spacetobatch_op_test_lib",
        ":sparse_to_dense_op_test_lib",
        # go/keep-sorted end
    ],
)

tf_xla_combined_py_test(
    name = "combined_ops_test_e",
    size = "medium",
    timeout = "long",
    package = "machina.compiler.tests",
    tags = [
        "no_cuda_asan",  # times out in individual tests
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "noasan",  # times out consistently from 2023-08-24
    ],
    tests = [
        # go/keep-sorted start
        ":adagrad_da_test_lib",
        ":adam_test_lib",
        ":argminmax_test_lib",
        ":listdiff_op_test_lib",
        ":slice_ops_test_lib",
        ":unary_ops_test_lib",
        # go/keep-sorted end
    ],
)

tf_xla_combined_py_test(
    name = "combined_ops_test_f",
    size = "medium",
    timeout = "long",
    # copybara:uncomment_begin
    # #TODO(b/286470564): Remove once the bug is fixed.
    # disable_tpu_tfrt = True,
    # copybara:uncomment_end
    package = "machina.compiler.tests",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    tests = [
        # go/keep-sorted start
        ":add_n_test_lib",
        ":cond_test_lib",
        ":while_test_lib",
        # go/keep-sorted end
    ],
)
#LINT.ThenChange(:individual_tests)

#LINT.IfChange(individual_tests)
tf_xla_py_strict_test(
    name = "adadelta_test",
    size = "medium",
    srcs = ["adadelta_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:adadelta",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "adagrad_test",
    size = "small",
    srcs = ["adagrad_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",  # Handled in the combined test above.
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:adagrad",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "adagrad_da_test",
    size = "small",
    srcs = ["adagrad_da_test.py"],
    tags = [
        "no_cuda_asan",  # times out
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:adagrad_da",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "adam_test",
    size = "small",
    srcs = ["adam_test.py"],
    tags = [
        "no_cuda_asan",  # times out
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:adam",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "add_n_test",
    size = "small",
    srcs = ["add_n_test.py"],
    # copybara:uncomment_begin
    # #TODO(b/286470564): Remove once the bug is fixed.
    # disable_tpu_tfrt = True,
    # copybara:uncomment_end
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "argminmax_test",
    size = "small",
    srcs = ["argminmax_test.py"],
    tags = [
        "no_cuda_asan",  # times out
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "noasan",  # times out consistently from 2023-08-24
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "binary_ops_test",
    size = "medium",
    srcs = ["binary_ops_test.py"],
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:bitwise_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "complex_div_test",
    size = "medium",
    srcs = ["complex_div_test.py"],
    enabled_backends = [
        "cpu",
        "gpu",
        "gpu_a100",
        "gpu_h100",
    ],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "bucketize_op_test",
    size = "small",
    srcs = ["bucketize_op_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "categorical_op_test",
    size = "small",
    srcs = ["categorical_op_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:random_seed",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "cholesky_op_test",
    size = "medium",
    srcs = ["cholesky_op_test.py"],
    disabled_backends = ["gpu_h100"],  # TODO: b/362907482 - Re-enable after fixing.
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "no_rocm",
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "cond_test",
    size = "small",
    srcs = ["cond_test.py"],
    # copybara:uncomment_begin
    # #TODO(b/286470564): Remove once the bug is fixed.
    # disable_tpu_tfrt = True,
    # copybara:uncomment_end
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/client:session",
        "//machina/python/compiler/xla",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:control_flow_switch_case",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "self_adjoint_eig_op_test",
    size = "medium",
    srcs = ["self_adjoint_eig_op_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "searchsorted_op_test",
    size = "small",
    timeout = "moderate",
    srcs = ["searchsorted_op_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "svd_op_test",
    size = "medium",
    srcs = ["svd_op_test.py"],
    disabled_backends = [
        # TODO(b/129396575): Fails on CPU.
        "cpu",
        "cpu_ondemand",
    ],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "no_rocm",
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:linalg_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "matrix_inverse_op_test",
    size = "small",
    timeout = "moderate",
    srcs = ["matrix_inverse_op_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "matrix_solve_op_test",
    size = "small",
    timeout = "moderate",
    srcs = ["matrix_solve_op_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:sysconfig",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "matrix_triangular_solve_op_test",
    size = "small",
    timeout = "moderate",
    srcs = ["matrix_triangular_solve_op_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "clustering_test",
    size = "small",
    srcs = ["clustering_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "concat_ops_test",
    size = "medium",
    srcs = ["concat_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "noasan",  # Timed out on 2023-07-12
    ],
    deps = [
        ":xla_test",
        "//machina/python:pywrap_sanitizers",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "conv2d_test",
    size = "medium",
    srcs = ["conv2d_test.py"],
    shard_count = 10,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":test_utils",
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "tensor_float_32_test",
    size = "medium",
    srcs = ["tensor_float_32_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    use_xla_device = False,  # Uses tf.function(jit_compile=True)
    deps = [
        ":xla_test",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:sysconfig",
        "//machina/python/platform:test",
    ],
)

tf_xla_py_strict_test(
    name = "conv3d_test",
    size = "medium",
    srcs = ["conv3d_test.py"],
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":test_utils",
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "depthwise_conv_op_test",
    size = "medium",
    srcs = ["depthwise_conv_op_test.py"],
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "no_rocm",
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_impl",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "dynamic_slice_ops_test",
    size = "small",
    srcs = ["dynamic_slice_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/compiler/tf2xla/python:xla",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "einsum_op_test",
    size = "medium",
    srcs = ["einsum_op_test.py"],
    enabled_backends = [
        "cpu",
        "gpu",
        "gpu_a100",
        "gpu_h100",
    ],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:special_math_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "reshape_op_test",
    size = "small",
    srcs = ["reshape_op_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "dynamic_stitch_test",
    size = "small",
    srcs = ["dynamic_stitch_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "extract_image_patches_op_test",
    size = "small",
    srcs = ["extract_image_patches_op_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "eager_test",
    size = "medium",
    srcs = ["eager_test.py"],
    tags = [
        "multi_and_single_gpu",
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//machina/core:protos_all_py",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/layers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:cond",
        "//machina/python/ops:embedding_ops",
        "//machina/python/ops:functional_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops_gen",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:test",
        "//machina/python/training:adam",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "fifo_queue_test",
    size = "medium",
    srcs = ["fifo_queue_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python:extra_py_tests_deps",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "fft_test",
    size = "medium",
    srcs = ["fft_test.py"],
    shard_count = 12,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "no_rocm",
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python:extra_py_tests_deps",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops/signal",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "slice_ops_test",
    size = "medium",
    srcs = ["slice_ops_test.py"],
    tags = [
        "no_cuda_asan",  # times out
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
    ],
)

tf_xla_py_strict_test(
    name = "ftrl_test",
    size = "medium",
    srcs = ["ftrl_test.py"],
    shard_count = 8,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:adagrad",
        "//machina/python/training:ftrl",
        "//machina/python/training:gradient_descent",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "ftrl_ops_test",
    size = "medium",
    srcs = ["ftrl_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:training_ops_gen",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "function_test",
    size = "small",
    srcs = ["function_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:function",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "image_ops_test",
    size = "small",
    timeout = "long",
    srcs = ["image_ops_test.py"],
    enabled_backends = [
        "cpu",
        "gpu",
        "gpu_a100",
        "gpu_h100",
    ],
    shard_count = 10,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",  # Times out frequently in fastbuild mode.
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:image_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "listdiff_op_test",
    size = "small",
    srcs = ["listdiff_op_test.py"],
    tags = [
        "no_cuda_asan",  # times out
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "lrn_ops_test",
    size = "medium",
    srcs = ["lrn_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "manip_ops_test",
    size = "small",
    srcs = ["manip_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:manip_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "matrix_band_part_test",
    size = "medium",
    timeout = "long",
    srcs = ["matrix_band_part_test.py"],
    tags = [
        "no_aarch64",  # TODO(b/315533266)
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "matrix_diag_ops_test",
    size = "medium",
    timeout = "long",
    srcs = ["matrix_diag_ops_test.py"],
    shard_count = 4,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "momentum_test",
    size = "small",
    srcs = ["momentum_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:momentum",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "nary_ops_test",
    size = "small",
    srcs = ["nary_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "nullary_ops_test",
    size = "small",
    srcs = ["nullary_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "pooling_ops_test",
    size = "medium",
    srcs = ["pooling_ops_test.py"],
    shard_count = 20,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "pooling_ops_3d_test",
    size = "medium",
    srcs = ["pooling_ops_3d_test.py"],
    shard_count = 20,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "proximal_adagrad_test",
    size = "medium",
    srcs = ["proximal_adagrad_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",  # Handled in the combined test above.
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:adagrad",
        "//machina/python/training:proximal_adagrad",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "proximal_gradient_descent_test",
    size = "medium",
    srcs = ["proximal_gradient_descent_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",  # Handled in the combined test above.
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:gradient_descent",
        "//machina/python/training:proximal_gradient_descent",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "qr_op_test",
    size = "medium",
    srcs = ["qr_op_test.py"],
    disabled_backends = [
        # Test is very slow on CPU.
        "cpu",
        "cpu_ondemand",
    ],
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "nozapfhahn",
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "unstack_test",
    size = "medium",
    srcs = ["unstack_test.py"],
    disabled_backends = [
        # TODO(b/149750262): timeout on CPU.
        "cpu",
        "cpu_ondemand",
    ],
    shard_count = 1,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "random_ops_test",
    size = "medium",
    srcs = ["random_ops_test.py"],
    shard_count = 10,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops/distributions:special_math",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "reduce_ops_test",
    size = "medium",
    srcs = ["reduce_ops_test.py"],
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "reduce_window_test",
    size = "small",
    srcs = ["reduce_window_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/compiler/tf2xla/python:xla",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:function",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "reverse_ops_test",
    size = "medium",
    srcs = ["reverse_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "reverse_sequence_op_test",
    size = "medium",
    srcs = ["reverse_sequence_op_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

# copybara:uncomment_begin(google-only)
# tf_xla_py_strict_test(
#     name = "reverse_sequence_op_args_test",
#     size = "medium",
#     srcs = ["reverse_sequence_op_args_test.py"],
#     main = "reverse_sequence_op_args_test.py",
#     tags = [
#         "no_pip",
#         "optonly",
#     ],
#     deps = [
#         ":xla_test",
#         "//machina/compiler/jit:xla_cpu_jit",  # DisableOnExport
#         "//machina/python/compat:v2_compat",
#         "//machina/python/eager:def_function",
#         "//machina/python/framework:errors",
#         "//machina/python/ops:array_ops",
#         "//machina/python/platform:client_testlib",
#     ],
# )
# copybara:uncomment_end

tf_xla_py_strict_test(
    name = "rmsprop_test",
    size = "small",
    srcs = ["rmsprop_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",  # Handled in the combined test above.
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:rmsprop",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "scan_ops_test",
    size = "medium",
    srcs = ["scan_ops_test.py"],
    shard_count = 6,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "nozapfhahn",
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "segment_reduction_ops_test",
    size = "medium",
    srcs = ["segment_reduction_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/client:device_lib",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "spacetobatch_op_test",
    size = "medium",
    srcs = ["spacetobatch_op_test.py"],
    shard_count = 3,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "sparse_to_dense_op_test",
    size = "medium",
    srcs = ["sparse_to_dense_op_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "stack_ops_test",
    size = "small",
    srcs = ["stack_ops_test.py"],
    enabled_backends = [
        "cpu",
        "gpu",
        "gpu_a100",
        "gpu_h100",
    ],
    tags = [
        "config-cuda-only",
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    use_xla_device = False,
    deps = [
        ":xla_test",
        "//machina/python/compiler/xla",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:data_flow_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "stateful_random_ops_test",
    size = "medium",
    srcs = ["stateful_random_ops_test.py"],
    disabled_backends = [
        "gpu",
        "gpu_a100",
        "gpu_h100",
    ],
    shard_count = 10,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/client",
        "//machina/python/client:device_lib",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/kernel_tests/random:util",
        "//machina/python/ops:random_ops_util",
        "//machina/python/ops:stateful_random_ops",
        "//machina/python/ops:stateful_random_ops_gen",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:flags",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "stateless_random_ops_test",
    size = "medium",
    srcs = ["stateless_random_ops_test.py"],
    disabled_backends = [
        "gpu",
        "gpu_a100",
        "gpu_h100",
    ],
    shard_count = 10,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "nozapfhahn",  # Times out under coverage
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/client:device_lib",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/kernel_tests/random:util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops_util",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops:stateless_random_ops_v2_gen",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "stochastic_cast_op_test",
    size = "medium",
    srcs = ["stochastic_cast_op_test.py"],
    # TODO(b/232442915): Enable GPUs.
    disabled_backends = [
        "gpu",
        "gpu_a100",
        "gpu_h100",
    ],
    shard_count = 20,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/kernel_tests/random:util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:standard_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops:stochastic_cast_op",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "tensor_array_ops_test",
    size = "medium",
    srcs = ["tensor_array_ops_test.py"],
    enabled_backends = [
        "cpu",
        "gpu",
        "gpu_a100",
        "gpu_h100",
    ],
    tags = [
        "config-cuda-only",
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "v1only",
    ],
    use_xla_device = False,
    deps = [
        ":xla_test",
        "//machina/python/compiler/xla",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_util",
        "//machina/python/ops:data_flow_ops_gen",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:tensor_array_grad",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "tensor_list_ops_test",
    size = "small",
    srcs = ["tensor_list_ops_test.py"],
    # copybara:uncomment_begin
    # #TODO(b/286470564): Remove once the bug is fixed.
    # disable_tpu_tfrt = True,
    # copybara:uncomment_end
    # TensorList ops are not implemented in the on-demand compilation model yet.
    disabled_backends = ["cpu_ondemand"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:list_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "ternary_ops_test",
    size = "medium",
    srcs = ["ternary_ops_test.py"],
    shard_count = 4,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//machina/python:extra_py_tests_deps",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "unary_ops_test",
    size = "medium",
    srcs = ["unary_ops_test.py"],
    shard_count = 20,
    tags = [
        "no_cuda_asan",  # times out
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "noasan",  #times out
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:bitwise_ops",
        "//machina/python/ops:functional_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

tf_xla_py_strict_test(
    name = "fused_batchnorm_test",
    size = "medium",
    srcs = ["fused_batchnorm_test.py"],
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":test_utils",
        ":xla_test",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:nn",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "variable_ops_test",
    size = "small",
    timeout = "moderate",
    srcs = ["variable_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:state_ops_gen",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:test",
        "//machina/python/training:gradient_descent",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "while_test",
    size = "small",
    srcs = ["while_test.py"],
    # copybara:uncomment_begin
    # #TODO(b/291130193): Remove once the bug is fixed.
    # disable_tpu_tfrt = True,
    # copybara:uncomment_end
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/compiler/tf2xla/python:xla",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:function",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "case_test",
    size = "small",
    srcs = ["case_test.py"],
    disabled_backends = ["cpu_ondemand"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    use_xla_device = False,  # Uses tf.function(jit_compile=True)
    deps = [
        ":xla_test",
        "//machina/python/eager:def_function",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_switch_case",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:io_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "cast_ops_test",
    size = "small",
    srcs = ["cast_ops_test.py"],
    disabled_backends = ["cpu_ondemand"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    use_xla_device = False,  # Uses tf.function(jit_compile=True)
    deps = [
        ":xla_test",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "gather_test",
    size = "medium",
    srcs = ["gather_test.py"],
    tags = [
        "no_pip",
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:flags",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "gather_nd_op_test",
    size = "medium",
    srcs = ["gather_nd_op_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "scatter_nd_op_test",
    size = "medium",
    srcs = ["scatter_nd_op_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "sort_ops_test",
    size = "medium",
    srcs = ["sort_ops_test.py"],
    shard_count = 50,
    # Times out in fastbuild mode.
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "nozapfhahn",
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/compiler/tf2xla/python:xla",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:function",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:sort_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "data_format_ops_test",
    size = "small",
    srcs = ["data_format_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "xla_device_test",
    size = "small",
    srcs = ["xla_device_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "fake_quant_ops_test",
    size = "medium",
    srcs = ["fake_quant_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "placeholder_test",
    size = "small",
    srcs = ["placeholder_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ],
    deps = [
        ":xla_test",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:test",
    ],
)

tf_xla_py_strict_test(
    name = "quantized_ops_test",
    size = "medium",
    srcs = ["quantized_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",  # Handled in the combined test above.
    ],
    deps = [
        ":xla_test",
        "//machina/compiler/tf2xla/python:xla",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:bitwise_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "xla_ops_test",
    size = "medium",
    srcs = ["xla_ops_test.py"],
    disabled_backends = [
        "gpu",
        "gpu_a100",
        "gpu_h100",
    ],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//machina/compiler/tf2xla/ops:gen_xla_ops",
        "//machina/compiler/tf2xla/python:xla",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:function",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:random_ops_util",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@local_xla//xla:xla_data_proto_py",
    ],
)

tf_xla_py_strict_test(
    name = "xla_custom_call_ops_test",
    size = "small",
    srcs = ["xla_custom_call_ops_test.py"],
    disabled_backends = [
        "gpu",
        "gpu_a100",
        "gpu_h100",
    ],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",  # Handled in the combined test above.
    ],
    use_xla_device = False,  # Uses tf.function(jit_compile=True)
    deps = [
        ":xla_test",
        "//machina/compiler/tf2xla/python:xla",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "runtime_shape_check_test",
    size = "small",
    srcs = ["runtime_shape_check_test.py"],
    disabled_backends = [
        "cpu",
        "cpu_ondemand",
    ],
    tags = [
        "no_pip",
        "notap",
    ],
    use_xla_device = False,
    deps = [
        ":xla_test",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "conv_node_name_test",
    size = "medium",
    srcs = ["conv_node_name_test.py"],
    tags = [
        "no_oss",  # TODO(b/148108508): Re-enable this test in OSS.
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:ops",
        "//machina/python/layers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "tridiagonal_solve_ops_test",
    size = "medium",
    srcs = ["tridiagonal_solve_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients",
        "//machina/python/ops:math_ops",
        "//machina/python/ops/linalg:linalg_impl",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "tridiagonal_matmul_ops_test",
    size = "medium",
    srcs = ["tridiagonal_matmul_ops_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops/linalg:linalg_impl",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tf_xla_py_strict_test(
    name = "special_math_test",
    size = "medium",
    srcs = ["special_math_test.py"],
    shard_count = 5,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python:extra_py_tests_deps",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:random_ops_gen",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "repeat_op_test",
    size = "medium",
    srcs = ["repeat_op_test.py"],
    shard_count = 1,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python:extra_py_tests_deps",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "image_ops_jit_compile_test",
    size = "medium",
    srcs = ["image_ops_jit_compile_test.py"],
    disabled_backends = [
        "cpu_ondemand",
    ],
    shard_count = 1,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    use_xla_device = False,
    deps = [
        ":xla_test",
        "//machina/python:extra_py_tests_deps",
        "//machina/python/eager:backprop",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "ensure_shape_op_test",
    size = "medium",
    srcs = ["ensure_shape_op_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "where_op_test",
    size = "small",
    srcs = ["where_op_test.py"],
    enabled_backends = [
        "cpu",
        "gpu",
        "gpu_a100",
        "gpu_h100",
    ],
    tags = [
        "no_pip",
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/tpu:tpu_py",
    ],
)

tf_xla_py_strict_test(
    name = "where_op_tpu_test",
    size = "small",
    srcs = ["where_op_test.py"],
    args = [
        "--tpu_use_tfrt=true",
    ],
    disabled_backends = [
        "cpu",
        "cpu_ondemand",
        "gpu",
        "gpu_a100",
        "gpu_h100",
    ],
    main = "where_op_test.py",
    tags = [
        "no_pip",
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/tpu:tpu_py",
    ],
)

tf_xla_py_strict_test(
    name = "const_arg_test",
    size = "small",
    srcs = ["const_arg_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",  # Handled in the combined test above.
    ],
    deps = [
        ":xla_test",
        "//machina/compiler/tf2xla/python:xla",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
    ],
)

tf_xla_py_strict_test(
    name = "sharding_util_ops_test",
    srcs = ["sharding_util_ops_test.py"],
    disabled_backends = [
        "cpu",
        "cpu_ondemand",
        "gpu",
        "gpu_a100",
        "gpu_h100",
    ],
    shard_count = 10,
    tags = [
        "notap",
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:tpu_ops_gen",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "approx_topk_test",
    srcs = ["approx_topk_test.py"],
    disable_experimental = False,
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:variables",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "xla_call_module_test",
    size = "small",
    srcs = ["xla_call_module_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    use_xla_device = False,  # Uses tf.function(jit_compile=True)
    deps = [
        ":xla_test",
        "//machina/compiler/mlir/stablehlo",
        "//machina/compiler/tf2xla/ops:gen_xla_ops",
        "//machina/compiler/tf2xla/python:xla",
        "//machina/core:protos_all_py",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:function",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "xla_call_module_no_platform_check_test",
    size = "small",
    srcs = ["xla_call_module_test.py"],
    # cpu_ondemand overrides the TF_MACHINA_MACHINA_XLA_FLAGS
    disabled_backends = ["cpu_ondemand"],
    env = {"TF_MACHINA_MACHINA_XLA_FLAGS": "--tf_xla_call_module_disabled_checks=platform"},
    main = "xla_call_module_test.py",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    use_xla_device = False,  # Uses tf.function(jit_compile=True)
    deps = [
        ":xla_test",
        "//machina/compiler/mlir/stablehlo",
        "//machina/compiler/tf2xla/ops:gen_xla_ops",
        "//machina/compiler/tf2xla/python:xla",
        "//machina/core:protos_all_py",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:function",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "xla_call_module_no_shape_assertions_check_test",
    size = "small",
    srcs = ["xla_call_module_test.py"],
    disabled_backends = ["cpu_ondemand"],  # cpu_ondemand overrides the TF_MACHINA_MACHINA_XLA_FLAGS
    env = {"TF_MACHINA_MACHINA_XLA_FLAGS": "--tf_xla_call_module_disabled_checks=shape_assertions"},
    main = "xla_call_module_test.py",
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    use_xla_device = False,  # Uses tf.function(jit_compile=True)
    deps = [
        ":xla_test",
        "//machina/compiler/mlir/stablehlo",
        "//machina/compiler/tf2xla/ops:gen_xla_ops",
        "//machina/compiler/tf2xla/python:xla",
        "//machina/core:protos_all_py",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:function",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "bincount_op_test",
    size = "small",
    srcs = ["bincount_op_test.py"],
    shard_count = 1,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",  # Handled in the combined test above.
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:errors",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/platform:test",
    ],
)

tf_xla_py_strict_test(
    name = "unique_ops_test",
    size = "small",
    srcs = ["unique_ops_test.py"],
    enabled_backends = [
        "cpu",
        "gpu",
        "gpu_a100",
        "gpu_h100",
    ],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

tpu_py_strict_test(
    name = "mean_op_test",
    size = "small",
    srcs = ["mean_op_test.py"],
    disable_experimental = False,
    tags = [
        "notsan",  # timesout
    ],
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_xla_py_strict_test(
    name = "xla_dump_to_test",
    size = "medium",
    srcs = ["xla_dump_to_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "optonly",
    ],
    deps = [
        ":xla_test",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

# copybara:uncomment_begin(google-only)
# tf_xla_py_strict_test(
#     name = "xla_dump_to_sponge_test",
#     size = "medium",
#     srcs = ["xla_dump_to_sponge_test.py"],
#     tags = [
#         "optonly",
#     ],
#     deps = [
#         ":xla_test",
#         "//third_party/py/numpy",
#         "//machina/python/ops:array_ops",
#         "//machina/python/ops:math_ops",
#         "//machina/python/platform:test",
#     ],
# )
# copybara:uncomment_end
#LINT.ThenChange(:combined_tests)

cuda_py_strict_test(
    name = "xla_device_gpu_test",
    size = "small",
    srcs = ["xla_device_gpu_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "jit_test",
    size = "medium",
    srcs = ["jit_test.py"],
    #shard_count = 5,
    tags = [
        "no_cuda_asan",  # Times out.
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        ":test_utils",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/compiler/xla:compiler_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:function",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "async_comp_test",
    size = "medium",
    srcs = ["async_comp_test.py"],
    shard_count = 1,
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:function",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "dense_layer_test",
    size = "medium",
    srcs = ["dense_layer_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        ":test_utils",
        "//machina/core:protos_all_py",
        "//machina/python/compiler/xla:compiler_py",
        "//machina/python/framework:ops",
        "//machina/python/layers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cc_library(
    name = "randomized_tests_library",
    testonly = 1,
    srcs = ["randomized_tests.cc"],
    deps = [
        "//machina/compiler/jit",
        "//machina/compiler/jit:common",
        "//machina/compiler/jit:flags_headers",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:machina_opensource",
        "//machina/core:test",
        "//machina/core:testlib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/tsl/platform:status",
    ],
)

tf_cuda_cc_test(
    name = "randomized_tests",
    size = "medium",
    args = ["--tf_xla_test_use_mlir=false"],
    shard_count = 20,
    # This test is randomized, so only run it if explicitly requested.
    tags = [
        "manual",
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ] + tf_cuda_tests_tags(),
    deps = [":randomized_tests_library"],
)

tf_cuda_cc_test(
    name = "randomized_tests_mlir",
    size = "medium",
    args = ["--tf_xla_test_use_mlir=true"],
    shard_count = 20,
    # This test is randomized, so only run it if explicitly requested.
    tags = [
        "manual",
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "notap",
    ] + tf_cuda_tests_tags(),
    deps = [":randomized_tests_library"],
)

# Create a deterministic version of randomized_tests_mlir with fixed seed.
# This can be used in presubmit checks as it is no longer randomized.
tf_cuda_cc_test(
    name = "randomized_tests_mlir_seeded",
    size = "medium",
    args = [
        "--tf_xla_random_seed=200839030",
        "--tf_xla_test_use_mlir=true",
        "--tf_xla_test_device=GPU:0",
    ],
    shard_count = 20,
    tags = [
        "config-cuda-only",
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "no_rocm",  # ROCmSoftwarePlatform #958
        "noasan",  # TODO(b/201651800)
        "requires-gpu-nvidia",
    ] + tf_cuda_tests_tags(),
    deps = [":randomized_tests_library"],
)

# Create a deterministic version of randomized_tests with fixed seed.
# This can be used in presubmit checks as it is no longer randomized.
tf_cuda_cc_test(
    name = "randomized_tests_seeded",
    size = "medium",
    args = [
        "--tf_xla_random_seed=200839030",
        "--tf_xla_test_use_mlir=false",
        "--tf_xla_test_device=GPU:0",
    ],
    shard_count = 20,
    tags = [
        "config-cuda-only",
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
        "no_rocm",  # ROCmSoftwarePlatform #958
        "noasan",  # TODO(b/201651800)
        "requires-gpu-nvidia",
    ] + tf_cuda_tests_tags(),
    deps = [":randomized_tests_library"],
)

tf_cuda_cc_test(
    name = "unary_ops_composition_test",
    srcs = ["unary_ops_composition_test.cc"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ] + tf_cuda_tests_tags(),
    deps = [
        "//machina/cc:cc_ops",
        "//machina/compiler/jit",
        "//machina/compiler/jit:flags",
        "//machina/compiler/jit:xla_kernel_creator",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "//machina/core/kernels:ops_testutil",
        "@local_xla//xla/tsl/platform:status",
    ],
)

py_strict_library(
    name = "lstm",
    testonly = 1,
    srcs = ["lstm.py"],
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variable_v1",
        "@six_archive//:six",
    ],
)

cuda_py_strict_test(
    name = "lstm_test",
    srcs = ["lstm_test.py"],
    tags = [
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        ":lstm",
        ":xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

# An example of ahead-of-time compilation using tfcompile.  The
# lstm_layer_inference.pbtxt file was generated by running lstm_test
# --dump_graph_dir, and the config file was written by hand.
#
# Run the following to build a minimal benchmark of the computation on Android:
# $ bazel build -c opt --cxxopt='-std=c++11' --linkopt='-lm' \
#   --cpu=armeabi-v7a \
#   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
#   --crosstool_top=//external:android/crosstool \
#   //machina/compiler/tests:lstm_layer_inference_benchmark

#
# Currently the resulting binary size is ~190KB
tf_library(
    name = "lstm_layer_inference",
    testonly = 1,
    config = "lstm_layer_inference.config.pbtxt",
    cpp_class = "LSTMLayerInference",
    graph = "lstm_layer_inference.pbtxt",
    tags = ["manual"],
    tfcompile_flags = ["--xla_cpu_multi_thread_eigen=false"],
)

cuda_py_strict_test(
    name = "const_test",
    size = "small",
    srcs = ["const_test.py"],
    xla_enable_strict_auto_jit = False,
    xla_enabled = True,
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tpu_py_strict_test(
    name = "giant_const_op_test",
    srcs = [
        "giant_const_op_test.py",
    ],
    disable_experimental = True,
    # TODO(b/188995810): Add an optimization in MLIR importer to not
    # materialize giant splat constants.
    deps = [
        "//machina/python/distribute:tpu_strategy",
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/eager:def_function",
        "//machina/python/eager:remote",
        "//machina/python/eager:test",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/platform:flags",
        "//third_party/py/numpy",
    ],
)
