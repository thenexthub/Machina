load("@local_xla//xla/tsl/platform:build_config_root.bzl", "if_pywrap")
load("//machina:pytype.default.bzl", "pytype_strict_library")
load(
    "//machina:machina.default.bzl",
    "get_compatible_with_portable",
    "tf_py_strict_test",
    "tf_python_pybind_extension",
)
load("//machina/compiler/mlir/quantization/stablehlo:internal_visibility_allowlist.bzl", "internal_visibility_allowlist")

package_group(
    name = "internal_visibility_allowlist_package",
    packages = [
        "//machina/compiler/mlir/lite/...",
        "//machina/compiler/mlir/quantization/...",
        "//machina/compiler/mlir/tf2xla/transforms/...",
        "//machina/lite/...",
        "//third_party/cloud_tpu/inference_converter/...",  # TPU Inference Converter V1
    ] + internal_visibility_allowlist(),
)

package(
    # copybara:uncomment default_applicable_licenses = ["@stablehlo//:license"],
    default_visibility = [
        ":internal_visibility_allowlist_package",
        "//machina:__pkg__",
    ],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "quantization",
    srcs = ["quantization.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_quantization",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//machina/compiler/mlir/quantization/machina/python:py_function_lib_py",
        "//machina/compiler/mlir/quantization/machina/python:save_model",
        "//machina/core:protos_all_py",
    ],
)

# copybara:uncomment_begin(google-only)
# pytype_strict_library(
#     name = "quantize_model_test_base",
#     testonly = 1,
#     srcs = ["integration_test/quantize_model_test_base.py"],
#     tags = ["no_pip"],
#     visibility = ["//visibility:private"],
#     deps = [
#         "//third_party/py/mlir:ir",
#         "//third_party/py/mlir:stablehlo_dialect",
#         "//third_party/py/mlir/_mlir_libs:_mlirRegisterEverything",
#         "//third_party/py/numpy",
#         "//machina:machina_py",
#         "//machina/compiler/mlir/stablehlo",
#         "//machina/python/eager:def_function",
#         "//machina/python/framework:dtypes",
#         "//machina/python/framework:ops",
#         "//machina/python/framework:tensor_spec",
#         "//machina/python/module",
#         "//machina/python/ops:array_ops",
#         "//machina/python/ops:math_ops",
#         "//machina/python/ops:nn_ops",
#         "//machina/python/ops:variables",
#         "//machina/python/platform:client_testlib",
#         "//machina/python/platform:tf_logging",
#         "//machina/python/saved_model:load",
#         "//machina/python/saved_model:loader",
#         "//machina/python/saved_model:save",
#         "//machina/python/types:core",
#         "@absl_py//absl/testing:parameterized",
#     ],
# )
#
# tf_py_strict_test(
#     name = "quantize_model_test",
#     srcs = ["integration_test/quantize_model_test.py"],
#     shard_count = 50,  # Parallelize the test to avoid timeouts.
#     deps = [
#         ":quantization",
#         ":quantize_model_test_base",
#         "//machina/compiler/mlir/quantization/common/python:testing",
#         "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
#         "//machina/compiler/mlir/quantization/machina/python:representative_dataset",
#         "//machina/python/eager:def_function",
#         "//machina/python/framework:dtypes",
#         "//machina/python/framework:ops",
#         "//machina/python/framework:tensor_spec",
#         "//machina/python/framework:test_lib",
#         "//machina/python/module",
#         "//machina/python/ops:math_ops",
#         "//machina/python/ops:nn_ops",
#         "//machina/python/platform:client_testlib",
#         "//machina/python/saved_model:load",
#         "//machina/python/saved_model:save",
#         "//machina/python/saved_model:tag_constants",
#         "//machina/python/types:core",
#         "@absl_py//absl/testing:parameterized",
#     ],
# )
# copybara:uncomment_end

# This is a header-only target. The purpose of `pywrap_quantization_lib_*` targets is to expose only
# the symbols that are required by `pywrap_quantization` that translates them to python functions.
# The only intended use case of this library is by `pywrap_quantization`. Not letting
# `pywrap_quantization` directly depend on sub-libraries like `static_range_srq` and instead haiving
# a consolidated impl library `pywrap_quantization_lib_impl` allows the maintainers to avoid
# declaring multiple impl libraries to `libmachina_cc` and `lib_pywrap_machina_internal`,
# which is required to avoid ODR violations.
cc_library(
    name = "pywrap_quantization_lib_header_only",
    srcs = [],
    hdrs = ["pywrap_quantization_lib.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:private"],  # ONLY for `pywrap_quantization`.
    deps = [
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/machina/python:py_function_lib",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

# See the comments for `pywrap_quantization_lib_header_only`.
cc_library(
    name = "pywrap_quantization_lib_impl",
    srcs = ["pywrap_quantization_lib.cc"],
    hdrs = ["pywrap_quantization_lib.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//machina:__pkg__",  # For libmachina_cc.so.
        "//machina/python:__pkg__",  # For lib_pywrap_machina_internal.so.
    ],
    deps = [
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo/cc:config",
        "//machina/compiler/mlir/quantization/stablehlo/cc:static_range_ptq",
        "//machina/compiler/mlir/quantization/stablehlo/cc:weight_only_ptq",
        "//machina/compiler/mlir/quantization/machina/python:py_function_lib",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
    ],
)

tf_python_pybind_extension(
    name = "pywrap_quantization",
    srcs = ["pywrap_quantization.cc"],
    pytype_srcs = ["pywrap_quantization.pyi"],
    starlark_only = True,
    visibility = [
        "//machina/python:__pkg__",
    ],
    # Each dependency MUST be either header-only or exclusive.
    deps = [
        "//machina/compiler/mlir/quantization/machina/python:type_casters",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:import_status_module",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ] + if_pywrap(
        if_false = [":pywrap_quantization_lib_header_only"],
        if_true = [":pywrap_quantization_lib_impl"],
    ),
)
