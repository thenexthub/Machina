# Definitions are loaded separately so that copybara can pattern match (and modify) each definition.
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_py_strict_test(
    name = "filter_fusion_test",
    size = "medium",
    srcs = ["filter_fusion_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "grappler_test",
    size = "medium",
    srcs = ["grappler_test.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:combinations",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "map_and_batch_fusion_test",
    size = "small",
    srcs = ["map_and_batch_fusion_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "map_and_filter_fusion_test",
    size = "small",
    srcs = ["map_and_filter_fusion_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "map_fusion_test",
    size = "medium",
    srcs = ["map_fusion_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "map_parallelization_test",
    size = "small",
    srcs = ["map_parallelization_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "filter_parallelization_test",
    size = "medium",
    srcs = ["filter_parallelization_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:checkpoint_test_base",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:script_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "make_deterministic_test",
    size = "small",
    srcs = ["make_deterministic_test.py"],
    deps = [
        "//machina/python/compat:v2_compat",
        "//machina/python/data/experimental/ops:batching",
        "//machina/python/data/experimental/ops:interleave_ops",
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/data/ops:readers",
        "//machina/python/eager:def_function",
        "//machina/python/framework:combinations",
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:cond",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:script_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "noop_elimination_test",
    size = "small",
    srcs = ["noop_elimination_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:logging_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "optimization_test",
    size = "medium",
    srcs = ["optimization_test.py"],
    shard_count = 2,
    tags = ["nomsan"],  # Runs out of memory.
    deps = [
        "//machina/python/data/experimental/ops:batching",
        "//machina/python/data/experimental/ops:grouping",
        "//machina/python/data/experimental/ops:scan_ops",
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "shuffle_and_repeat_fusion_test",
    size = "small",
    srcs = ["shuffle_and_repeat_fusion_test.py"],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
    deps = [
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "seq_interleave_prefetch_test",
    size = "medium",
    srcs = ["seq_interleave_prefetch_test.py"],
    deps = [
        "//machina/python/data/experimental/ops:batching",
        "//machina/python/data/experimental/ops:grouping",
        "//machina/python/data/experimental/ops:scan_ops",
        "//machina/python/data/experimental/ops:testing",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:options",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
