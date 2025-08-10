# Tests of TensorFlow kernels written using the Python API.

load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

# CPU-only tests should use tf_py_test, GPU tests use cuda_py_test
# Please avoid the py_tests and cuda_py_tests (plural) while we
# fix the shared/overbroad dependencies.

cuda_py_strict_test(
    name = "benchmark_test",
    size = "small",
    srcs = ["benchmark_test.py"],
    tags = ["no_windows"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:benchmark",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
    ],
)

cuda_py_strict_test(
    name = "check_ops_test",
    size = "small",
    srcs = ["check_ops_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:gradients",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "collective_ops_multi_worker_test",
    size = "medium",
    srcs = ["collective_ops_multi_worker_test.py"],
    tags = [
        "no_mac",  # times out on CI but works fine locally
        "no_rocm",
    ],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/distribute:combinations",
        "//machina/python/distribute:multi_process_runner",
        "//machina/python/distribute:multi_worker_test_base",
        "//machina/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//machina/python/eager:context",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:collective_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

# b/234468872
cuda_py_strict_test(
    name = "collective_ops_test",
    size = "medium",
    srcs = ["collective_ops_test.py"],
    shard_count = 4,
    tags = [
        # "multi_and_single_gpu", # TODO(b/287692888): re-enable once the 2gpu test passes.
        "no_tfrt",  # TODO(b/185944042)
    ],
    deps = [
        "//machina/python/compat:v2_compat",
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute:combinations",
        "//machina/python/distribute:test_util",
        "//machina/python/eager:cancellation",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:collective_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "composite_tensor_ops_test",
    size = "small",
    srcs = ["composite_tensor_ops_test.py"],
    deps = [
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:composite_tensor_ops",
        "//machina/python/ops:composite_tensor_ops_gen",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:list_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:test",
        "//machina/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "critical_section_test",
    size = "medium",
    srcs = ["critical_section_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:prefetching_ops",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:control_flow_v2_toggles",
        "//machina/python/ops:critical_section_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "garbage_collection_test",
    size = "small",
    srcs = ["garbage_collection_test.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "gradient_correctness_test",
    size = "small",
    srcs = ["gradient_correctness_test.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "logging_ops_logging_level_test",
    size = "small",
    srcs = ["logging_ops_logging_level_test.py"],
    tags = [
        "no_oss",  # b/198486357
        "no_windows",
    ],
    deps = [
        "//machina/python/framework:test_lib",
        "//machina/python/ops:logging_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
    ],
)

cuda_py_strict_test(
    name = "logging_ops_test",
    size = "medium",
    srcs = ["logging_ops_test.py"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:logging_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "metrics_test",
    size = "medium",
    srcs = ["metrics_test.py"],
    shard_count = 20,
    tags = ["no_windows_gpu"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:data_flow_grad",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:metrics",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "numerics_test",
    size = "small",
    srcs = ["numerics_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:numerics",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "template_test",
    size = "small",
    srcs = ["template_test.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/framework:ops",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:template",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:gradient_descent",
    ],
)

cuda_py_strict_test(
    name = "trace_op_test",
    size = "small",
    srcs = ["trace_op_test.py"],
    tags = ["no_windows_gpu"],
    deps = [
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "histogram_ops_test",
    size = "small",
    srcs = ["histogram_ops_test.py"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:histogram_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
