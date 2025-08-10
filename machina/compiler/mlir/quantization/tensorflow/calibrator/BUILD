load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:pytype.default.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_library")

# Placeholder: load py_proto_library
load(
    "//machina:machina.bzl",
    "tf_cc_test",
    "tf_gen_op_wrapper_py",
)
load(
    "//machina:machina.default.bzl",
    "get_compatible_with_portable",
    "tf_kernel_library",
    "tf_py_strict_test",
)
load("//machina/core/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/compiler/mlir/quantization/machina:internal_visibility_allowlist_package",
        "//machina/core:__pkg__",
        "//machina/tools/pip_package:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "calibration_statistics_collector_base",
    hdrs = ["calibration_statistics_collector_base.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":calibration_statistics_proto_cc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "calibration_statistics_collector_min_max",
    srcs = ["calibration_statistics_collector_min_max.cc"],
    hdrs = ["calibration_statistics_collector_min_max.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":calibration_statistics_collector_base",
        ":calibration_statistics_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "calibration_statistics_collector_average_min_max",
    srcs = ["calibration_statistics_collector_average_min_max.cc"],
    hdrs = ["calibration_statistics_collector_average_min_max.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":calibration_statistics_collector_base",
        ":calibration_statistics_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "calibration_statistics_collector_histogram",
    srcs = ["calibration_statistics_collector_histogram.cc"],
    hdrs = ["calibration_statistics_collector_histogram.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":calibration_statistics_collector_base",
        ":calibration_statistics_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo/cc/calibration:calibration_parameters",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "@com_google_absl//absl/types:span",
    ],
)

pytype_strict_library(
    name = "calibration_algorithm",
    srcs = ["calibration_algorithm.py"],
    deps = [
        ":calibration_statistics_proto_py",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//third_party/py/numpy",
    ],
)

pytype_strict_contrib_test(
    name = "calibration_algorithm_test",
    srcs = ["calibration_algorithm_test.py"],
    deps = [
        ":calibration_algorithm",
        ":calibration_statistics_proto_py",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_proto_library(
    name = "calibration_statistics_proto",
    srcs = ["calibration_statistics.proto"],
    make_default_target_header_only = True,
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "calibration_statistics_py_pb2",
#     deps = [
#         ":calibration_statistics_proto",
#     ],
# )
# copybara:uncomment_end

tf_cc_test(
    name = "calibration_statistics_collector_test",
    size = "small",
    srcs = ["calibration_statistics_collector_test.cc"],
    deps = [
        ":calibration_statistics_collector_average_min_max",
        ":calibration_statistics_collector_histogram",
        ":calibration_statistics_collector_min_max",
        ":calibration_statistics_proto_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_kernel_library(
    name = "custom_aggregator_op",
    srcs = ["custom_aggregator_op.cc"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//machina:__pkg__",
        "//machina/compiler/mlir/quantization/machina/python:__pkg__",
    ],
    deps = [
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo/cc/calibration:calibration_parameters",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_custom_aggregator_op_wrapper",
    out = "custom_aggregator_op_wrapper.py",
    extra_py_deps = [
        "//machina/python:pywrap_tfe",
        "//machina/python/util:dispatch",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
    # Prevent unintentionally generating Python wrappers for all TF ops.
    op_allowlist = ["CustomAggregator"],
    py_lib_rule = py_strict_library,
    visibility = ["//visibility:private"],
    deps = [":custom_aggregator_op"],
)

tf_py_strict_test(
    name = "custom_aggregator_op_test",
    size = "small",
    srcs = ["integration_test/custom_aggregator_op_test.py"],
    deps = [
        ":calibration_statistics_proto_py",
        ":gen_custom_aggregator_op_wrapper",
        "//machina:machina_py",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//machina/python:pywrap_machina",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_kernel_library(
    name = "calibration_statistics_saver_op",
    srcs = ["calibration_statistics_saver_op.cc"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//machina/compiler/mlir/quantization/machina/python:__pkg__"],
    deps = [
        ":calibration_statistics_collector_average_min_max",
        ":calibration_statistics_collector_base",
        ":calibration_statistics_collector_histogram",
        ":calibration_statistics_collector_min_max",
        ":calibration_statistics_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:logging",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/platform:env",
    ],
)

tf_cc_test(
    name = "calibration_statistics_saver_op_test",
    srcs = ["calibration_statistics_saver_op_test.cc"],
    deps = [
        ":calibration_statistics_proto_cc",
        ":calibration_statistics_saver_op",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_googletest//:gtest",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:status_matchers",
    ],
)
