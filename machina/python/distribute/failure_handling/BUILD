load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "tf_custom_op_py_strict_library", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "failure_handling_lib",
    srcs = [
        "failure_handling.py",
    ],
    deps = [
        ":check_preemption_py",
        ":failure_handling_util",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint:checkpoint_context",
        "//machina/python/checkpoint:checkpoint_management",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:multi_worker_util",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:variables",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_contextlib",
        "//machina/python/util:tf_export",
        "//machina/tools/docs:doc_controls",
    ],
)

py_strict_library(
    name = "failure_handling_util",
    srcs = [
        "failure_handling_util.py",
    ],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/platform:tf_logging",
        "@pypi_requests//:pkg",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "preemption_watcher",
    srcs = ["preemption_watcher.py"],
    deps = [
        ":failure_handling_util",
        "//machina/python/eager:context",
        "//machina/python/eager:monitoring",
        "//machina/python/framework:errors",
        "//machina/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "failure_handler_test",
    timeout = "long",
    srcs = ["failure_handler_test.py"],
    shard_count = 8,
    tags = [
        "no_mac",
        "no_pip",  # TODO(b/266520226)
        "no_windows",  # TODO(b/197981388)
    ],
    deps = [
        ":failure_handling_lib",
        ":failure_handling_util",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint:checkpoint_management",
        "//machina/python/distribute:collective_all_reduce_strategy",
        "//machina/python/distribute:combinations",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:mirrored_strategy",
        "//machina/python/distribute:multi_process_runner",
        "//machina/python/distribute:multi_worker_test_base",
        "//machina/python/distribute:multi_worker_util",
        "//machina/python/distribute:one_device_strategy",
        "//machina/python/distribute:test_util",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/module",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:server_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "gce_failure_handler_test",
    srcs = ["gce_failure_handler_test.py"],
    shard_count = 32,
    tags = [
        "no_mac",  # Fails on CI but works fine locally.
        "no_pip",  # TODO(b/266520226)
        "noasan",  # TODO(b/226154233): Flaky test
    ],
    deps = [
        ":failure_handling_lib",
        ":failure_handling_util",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint:checkpoint_management",
        "//machina/python/distribute:collective_all_reduce_strategy",
        "//machina/python/distribute:combinations",
        "//machina/python/distribute:mirrored_strategy",
        "//machina/python/distribute:multi_process_runner",
        "//machina/python/distribute:multi_worker_test_base",
        "//machina/python/distribute:multi_worker_util",
        "//machina/python/distribute:one_device_strategy",
        "//machina/python/distribute:test_util",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/module",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:server_lib",
        "@absl_py//absl/testing:parameterized",
        "@pypi_dill//:pkg",  # build_cleaner: keep
    ],
)

tf_custom_op_py_strict_library(
    name = "check_preemption_py",
    kernels = [
        "//machina/core/distributed_runtime/preemption:check_preemption_op_kernel",
        "//machina/core/distributed_runtime/preemption:check_preemption_op_op_lib",
    ],
    deps = [
        "//machina/core/distributed_runtime/preemption:gen_check_preemption_op",
        "//machina/python/framework:for_generated_wrappers",
    ],
)
