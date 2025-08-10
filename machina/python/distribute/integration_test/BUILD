load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test")
load("//machina/core/platform:distribute.bzl", "distribute_py_strict_test")
load("//machina/python/tpu:tpu.bzl", "tpu_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

distribute_py_strict_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.py"],
    tags = [
        "no_windows",  # TODO(b/171350360)
        "nomultivm",  # multi_worker_test_base incompatible with multivm base
    ],
    deps = [
        "//machina:machina_py",
        "//machina/python/distribute:combinations",
        "//machina/python/distribute:multi_worker_test_base",
        "//machina/python/distribute:parameter_server_strategy_v2",
        "//machina/python/distribute:sharded_variable",
        "//machina/python/distribute:strategy_combinations",
        "//machina/python/distribute:test_util",
        "//machina/python/distribute:values",
        "//machina/python/eager:context",
        "//machina/python/eager:test",
        "//machina/python/framework:errors",
        "//machina/python/ops:lookup_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "mwms_peer_failure_test",
    size = "medium",
    srcs = ["mwms_peer_failure_test.py"],
    shard_count = 2,
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/227372713)
        "notsan",  # b/195248428
    ],
    deps = [
        "//machina:machina_py",
        "//machina/python/distribute:collective_all_reduce_strategy",
        "//machina/python/distribute:multi_process_runner",
        "//machina/python/distribute:multi_worker_test_base",
        "//machina/python/distribute:test_util",
        "//machina/python/eager:test",
    ],
)

py_strict_library(
    name = "mwms_peer_failure_test_lib",
    srcs = ["mwms_peer_failure_test.py"],
    visibility = ["//learning/brain/runtime/python:__pkg__"],
    deps = [
        "//machina:machina_py",
        "//machina/python/distribute:collective_all_reduce_strategy",
        "//machina/python/distribute:multi_process_runner",
        "//machina/python/distribute:multi_worker_test_base",
        "//machina/python/distribute:test_util",
        "//machina/python/eager:test",
    ],
)

tpu_py_strict_test(
    name = "tpu_memory_test",
    size = "medium",
    srcs = ["tpu_memory_test.py"],
    disable_experimental = True,
    disable_mlir_bridge = True,
    disable_tfrt = False,
    disable_v2 = True,
    tags = ["no_oss"],
    deps = [
        "//machina:machina_py",
        "//machina/python/eager:context",
        "//machina/python/platform:flags",
    ],
)
