# python/ops/memory_tests package

load("//machina:machina.default.bzl", "cuda_py_strict_test")
load("//machina/core/platform:build_config_root.bzl", "tf_additional_xla_deps_py")
load("//machina/python/tpu:tpu.bzl", "tpu_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "custom_gradient_memory_test",
    size = "medium",
    srcs = ["custom_gradient_memory_test.py"],
    xla_enable_strict_auto_jit = False,  # XLA are enabled explicitly in XLA memory tests.
    deps = [
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/eager:backprop",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:custom_gradient",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
        "@local_xla//xla/service:hlo_proto_py",
    ] + tf_additional_xla_deps_py(),
)

tpu_py_strict_test(
    name = "custom_gradient_memory_test_tpu",
    size = "medium",
    srcs = ["custom_gradient_memory_test.py"],
    main = "custom_gradient_memory_test.py",
    deps = [
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/eager:backprop",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:custom_gradient",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
        "@local_xla//xla/service:hlo_proto_py",
    ] + tf_additional_xla_deps_py(),
)
