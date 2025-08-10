load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "cross_device_ops_test",
    srcs = ["cross_device_ops_test.py"],
    tags = [
        # "multi_and_single_gpu", # TODO(b/287692888): re-enable once the 2gpu test passes.
        "no_windows_gpu",  # b/216367668
        "no_windows",  # b/326464742
    ],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/distribute:collective_all_reduce_strategy",
        "//machina/python/distribute:collective_util",
        "//machina/python/distribute:combinations",
        "//machina/python/distribute:cross_device_ops",
        "//machina/python/distribute:cross_device_utils",
        "//machina/python/distribute:device_util",
        "//machina/python/distribute:distribute_utils",
        "//machina/python/distribute:multi_worker_test_base",
        "//machina/python/distribute:multi_worker_util",
        "//machina/python/distribute:reduce_util",
        "//machina/python/distribute:strategy_combinations",
        "//machina/python/distribute:values",
        "//machina/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//machina/python/eager:context",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:kernels",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:collective_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "all_reduce",
    srcs = [
        "all_reduce.py",
    ],
    deps = [
        "//machina/python/framework:device",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nccl_ops",
    ],
)

tf_py_strict_test(
    name = "all_reduce_test",
    srcs = ["all_reduce_test.py"],
    deps = [
        ":all_reduce",
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "input_lib",
    srcs = ["input_lib.py"],
    deps = [
        "//machina/python/data/experimental/ops:cardinality",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:multi_device_iterator_ops",
        "//machina/python/data/ops:optional_ops",
        "//machina/python/distribute:input_lib",
        "//machina/python/eager:context",
        "//machina/python/framework:ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/types:data",
        "//machina/python/util:deprecation",
    ],
)
