load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

licenses(["notice"])

py_strict_library(
    name = "config",
    srcs = ["config.py"],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        ":config",
        ":extensions",
        "//machina:machina_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops/numpy_ops:np_array_ops",
        "//machina/python/ops/numpy_ops:np_utils",
        "//machina/python/ops/numpy_ops:numpy",
        "//machina/python/util:nest",
        "//machina/python/util:numpy_compat",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "np_wrapper",
    srcs = ["np_wrapper.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//machina/python/compat:v2_compat",
        "//machina/python/framework:dtypes",
        "//machina/python/ops/numpy_ops:np_array_ops",
        "//machina/python/ops/numpy_ops:np_arrays",
        "//machina/python/ops/numpy_ops:np_config",
        "//machina/python/ops/numpy_ops:np_dtypes",
        "//machina/python/ops/numpy_ops:np_math_ops",
        "//machina/python/ops/numpy_ops:np_random",
        "//machina/python/ops/numpy_ops:np_utils",
        "//machina/python/ops/numpy_ops:numpy",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "extensions",
    srcs = ["extensions.py"],
    deps = [
        ":np_wrapper",
        "//machina:machina_py",
        "//machina/python/compiler/xla",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager/polymorphic_function",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:device_spec",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_conversion_registry",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:bitwise_ops_gen",
        "//machina/python/ops:clip_ops",
        "//machina/python/ops:collective_ops_gen",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:custom_gradient",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:special_math_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/ops/numpy_ops:numpy",
        "//machina/python/ops/parallel_for:control_flow_ops",
        "//machina/python/tpu:tpu_py",
        "//machina/python/tpu/ops",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

# copybara:uncomment_begin(google-only)
# py_strict_test(
#     name = "extensions_test",
#     srcs = ["extensions_test.py"],
#     tags = [
#         "gpu",
#         "no_pip",
#         "notap",  # b/294137902
#         "requires-gpu-nvidia",
#     ],
#     deps = [
#         ":extensions",
#         ":np_wrapper",
#         "//learning/brain/research/jax:gpu_support",
#         # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
#         "//third_party/py/jax",
#         "//third_party/py/numpy",
#         "//machina:machina_py",
#         "//machina/python/distribute/cluster_resolver/tpu:tpu_cluster_resolver_py",
#         "//machina/python/eager:backprop",
#         "//machina/python/eager:context",
#         "//machina/python/eager/polymorphic_function",
#         "//machina/python/framework:config",
#         "//machina/python/framework:constant_op",
#         "//machina/python/framework:dtypes",
#         "//machina/python/framework:ops",
#         "//machina/python/framework:tensor",
#         "//machina/python/framework:tensor_shape",
#         "//machina/python/ops:array_ops",
#         "//machina/python/ops:gradient_checker_v2",
#         "//machina/python/ops:math_ops",
#         "//machina/python/ops:nn_ops",
#         "//machina/python/ops:random_ops",
#         "//machina/python/ops:stateful_random_ops",
#         "//machina/python/ops/numpy_ops:numpy",
#         "//machina/python/platform:client_testlib",
#         "//machina/python/util:nest",
#         "@absl_py//absl/flags",
#         "@absl_py//absl/testing:parameterized",
#     ],
# )
#
# py_strict_test(
#     name = "extensions_test_tpu",
#     srcs = ["extensions_test.py"],
#     args = [
#         "--jax_allow_unused_tpus",
#         "--requires_tpu",
#     ],
#     main = "extensions_test.py",
#     tags = [
#         "no_pip",
#         "requires-tpu",
#     ],
#     deps = [
#         ":extensions",
#         ":np_wrapper",
#         "//learning/brain/google/xla",
#         # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
#         "//third_party/py/jax",
#         "//third_party/py/numpy",
#         "//machina:machina_py",
#         "//machina/python/distribute/cluster_resolver/tpu:tpu_cluster_resolver_py",
#         "//machina/python/eager:backprop",
#         "//machina/python/eager:context",
#         "//machina/python/eager/polymorphic_function",
#         "//machina/python/framework:config",
#         "//machina/python/framework:constant_op",
#         "//machina/python/framework:dtypes",
#         "//machina/python/framework:ops",
#         "//machina/python/framework:tensor",
#         "//machina/python/framework:tensor_shape",
#         "//machina/python/ops:array_ops",
#         "//machina/python/ops:gradient_checker_v2",
#         "//machina/python/ops:math_ops",
#         "//machina/python/ops:nn_ops",
#         "//machina/python/ops:random_ops",
#         "//machina/python/ops:stateful_random_ops",
#         "//machina/python/ops/numpy_ops:numpy",
#         "//machina/python/platform:client_testlib",
#         "//machina/python/util:nest",
#         "@absl_py//absl/flags",
#         "@absl_py//absl/testing:parameterized",
#     ],
# )
# copybara:uncomment_end

py_strict_test(
    name = "np_test",
    timeout = "long",
    srcs = ["np_test.py"],
    args = [
        "--num_generated_cases=90",
        "--enable_x64",  # Needed to enable dtype check
    ],
    shard_count = 20,
    tags = [
        "gpu",
        "no_pip",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":config",
        ":extensions",
        ":np_wrapper",
        ":test_util",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops/numpy_ops:np_config",
        "//machina/python/ops/numpy_ops:numpy",
        "//machina/python/util:nest",
        "//machina/python/util:numpy_compat",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "np_indexing_test",
    srcs = ["np_indexing_test.py"],
    args = [
        "--num_generated_cases=90",
        "--enable_x64",  # Needed to enable dtype check
    ],
    shard_count = 10,
    # TODO(b/164245103): Re-enable GPU once tf.tensor_strided_slice_update's segfault is fixed.
    tags = [
        "no_pip",
        #     "gpu",
        #     "requires-gpu-nvidia",
    ],
    deps = [
        ":config",
        ":extensions",
        ":np_wrapper",
        ":test_util",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//machina/python/framework:config",
        "//machina/python/ops/numpy_ops:numpy",
        "//machina/python/util:nest",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "np_einsum_test",
    srcs = ["np_einsum_test.py"],
    args = [
        "--num_generated_cases=90",
        "--enable_x64",  # Needed to enable dtype check
    ],
    shard_count = 20,
    tags = [
        "gpu",
        "no_pip",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":config",
        ":np_wrapper",
        ":test_util",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//machina/python/ops/numpy_ops:np_config",
        "//machina/python/ops/numpy_ops:numpy",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
