load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//machina:machina.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "loss_scale",
    srcs = ["loss_scale.py"],
    deps = [
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:reduce_util",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/trackable:base",
        "//machina/python/util:deprecation",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "loss_scale_optimizer",
    srcs = ["loss_scale_optimizer.py"],
    deps = [
        ":loss_scale",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:smart_cond",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/training:optimizer",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_test(
    name = "loss_scale_optimizer_test",
    size = "small",
    srcs = ["loss_scale_optimizer_test.py"],
    deps = [
        ":loss_scale",
        ":loss_scale_optimizer",
        "//machina/python/checkpoint",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:mirrored_strategy",
        "//machina/python/eager:context",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:custom_gradient",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:gradient_descent",
        "//machina/python/training:momentum",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "loss_scale_test",
    size = "medium",
    srcs = ["loss_scale_test.py"],
    deps = [
        ":loss_scale",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute:distribute_lib",
        "//machina/python/distribute:mirrored_strategy",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "mixed_precision_global_state",
    srcs = ["mixed_precision_global_state.py"],
    deps = ["//machina/python/util:tf_export"],
)

py_strict_library(
    name = "mixed_precision",
    srcs = ["mixed_precision.py"],
    deps = [
        ":loss_scale_optimizer",
        ":mixed_precision_global_state",
        "//machina/python/framework:config",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:optimizer",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "mixed_precision_test",
    size = "small",
    srcs = ["mixed_precision_test.py"],
    deps = [
        ":loss_scale_optimizer",
        ":mixed_precision",
        ":mixed_precision_global_state",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:gradient_descent",
        "@absl_py//absl/testing:parameterized",
    ],
)
