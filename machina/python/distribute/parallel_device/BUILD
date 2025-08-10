load("//machina:strict.default.bzl", "py_strict_library")
load("//machina/core/platform:distribute.bzl", "distribute_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

# Pybind rules must live in machina/python due to header rule visibility.
exports_files(
    ["pywrap_parallel_device.cc"],
    visibility = ["//machina/python:__pkg__"],
)

py_strict_library(
    name = "parallel_device",
    srcs = ["parallel_device.py"],
    visibility = [
        "//machina:internal",
        "//third_party/py/machina_numerics/extensions:__pkg__",
    ],
    deps = [
        "//machina/python:_pywrap_parallel_device",
        "//machina/python/distribute:device_util",
        "//machina/python/eager:context",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:variables",
        "//machina/python/tpu/ops",
        "//machina/python/util:nest",
        "//machina/python/util:variable_utils",
    ],
)

distribute_py_strict_test(
    name = "parallel_device_test",
    srcs = ["parallel_device_test.py"],
    # We don't only use TPU Runtime on parallel device, we use c_api_tfrt instead.
    disable_tpu_use_tfrt = True,
    tags = [
        # Dependencies aren't otherwise included in the pip package yet.
        "no_pip",
        "notpu",  # TODO(b/210159281)
    ],
    deps = [
        ":parallel_device",
        "//machina/python/checkpoint",
        "//machina/python/checkpoint:checkpoint_management",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:collective_ops",
        "//machina/python/ops:control_flow_switch_case",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:stateful_random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:save",
        "//machina/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)
