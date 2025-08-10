load("//machina:strict.default.bzl", "py_strict_library")
load(
    "//machina/dtensor:build_defs.bzl",
    "GPU_2DEVS_BACKEND",
    "TPU_V3_DONUT_BACKEND",
    "dtensor_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//learning/brain/distribute/python/benchmark:__subpackages__",
        "//machina:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "experimental",
    srcs = [
        "__init__.py",
    ],
    deps = [
        ":mirrored_strategy",
        ":multi_worker_mirrored_strategy",
    ],
)

py_strict_library(
    name = "mirrored_strategy",
    srcs = ["mirrored_strategy.py"],
    deps = [
        ":dtensor_strategy_extended",
        ":dtensor_util",
        "//machina/dtensor/python:config",
        "//machina/dtensor/python:mesh_util",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/framework:device",
    ],
)

dtensor_test(
    name = "mirrored_strategy_test",
    srcs = ["mirrored_strategy_test.py"],
    shard_count = {"tpu": 2},
    tags = ["no_pip"],
    deps = [
        ":dtensor_util",
        ":mirrored_strategy",
        "//machina/dtensor/python:api",
        "//machina/dtensor/python:d_variable",
        "//machina/dtensor/python:layout",
        "//machina/dtensor/python:mesh_util",
        "//machina/dtensor/python/tests:test_util",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:reduce_util",
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops:variables",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "dtensor_util",
    srcs = ["dtensor_util.py"],
    deps = [
        "//machina/dtensor/python:accelerator_util",
        "//machina/dtensor/python:api",
        "//machina/dtensor/python:input_util",
        "//machina/dtensor/python:layout",
        "//machina/python/distribute:cross_device_ops",
        "//machina/python/distribute:device_util",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:reduce_util",
        "//machina/python/distribute:values",
        "//machina/python/eager:context",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion_registry",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:summary_ops_v2",
    ],
)

dtensor_test(
    name = "dtensor_util_test",
    srcs = ["dtensor_util_test.py"],
    tags = ["no_pip"],
    deps = [
        ":dtensor_util",
        ":mirrored_strategy",
        "//machina/dtensor/python:api",
        "//machina/dtensor/python:layout",
        "//machina/dtensor/python/tests:test_util",
        "//machina/python/distribute:reduce_util",
        "//machina/python/distribute:values",
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
        "//machina/python/framework:constant_op",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "dtensor_strategy_extended",
    srcs = ["dtensor_strategy_extended.py"],
    deps = [
        ":dtensor_util",
        "//machina/dtensor/python:api",
        "//machina/dtensor/python:config",
        "//machina/dtensor/python:d_variable",
        "//machina/dtensor/python:input_util",
        "//machina/dtensor/python:layout",
        "//machina/python/data/experimental/ops:distribute",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:distribute_utils",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/util:nest",
    ],
)

py_strict_library(
    name = "multi_worker_mirrored_strategy",
    srcs = ["multi_worker_mirrored_strategy.py"],
    deps = [
        ":dtensor_strategy_extended",
        ":dtensor_util",
        "//machina/dtensor/python:config",
        "//machina/dtensor/python:mesh_util",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:multi_worker_util",
        "//machina/python/distribute/cluster_resolver:tfconfig_cluster_resolver_py",
    ],
)

dtensor_test(
    name = "multi_worker_mirrored_strategy_test",
    srcs = ["multi_worker_mirrored_strategy_test.py"],
    additional_backends = [
        GPU_2DEVS_BACKEND,
        TPU_V3_DONUT_BACKEND,
    ],
    disable = [
        "gpu",  # multi-client gpu is tested via GPU_2DEVS_BACKEND.
        "tpu",  # multi-client tpu is tested via TPU_V3_DONUT_BACKEND.
    ],
    disable_tfrt = [
        "cpu",  # TODO(b/217969210): Re-enable in TFRT CPU.
        GPU_2DEVS_BACKEND,  # TODO(b/230679405): Re-enable in TFRT GPU.
    ],
    tags = [
        "no_pip",
        "no_windows",
        "nosan",
    ],  # b/195537906
    deps = [
        ":dtensor_util",
        ":multi_worker_mirrored_strategy",
        "//machina/dtensor/python:api",
        "//machina/dtensor/python:d_variable",
        "//machina/dtensor/python:layout",
        "//machina/dtensor/python/tests:multi_client_test_util",
        "//machina/dtensor/python/tests:test_util",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:reduce_util",
        "//machina/python/distribute/cluster_resolver:tfconfig_cluster_resolver_py",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:stateless_random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)
