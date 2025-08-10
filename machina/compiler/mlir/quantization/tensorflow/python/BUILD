load("@local_xla//xla/tsl/platform:build_config_root.bzl", "if_pywrap")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:pytype.default.bzl", "pytype_strict_library")
load(
    "//machina:machina.default.bzl",
    "get_compatible_with_portable",
    "tf_py_strict_test",
    "tf_python_pybind_extension",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/compiler/mlir/quantization/stablehlo:__subpackages__",
        "//machina/compiler/mlir/quantization/machina:internal_visibility_allowlist_package",
        "//machina/lite:__subpackages__",
        "//machina/python:__subpackages__",
        "//machina/tools/pip_package:__subpackages__",
    ],
    licenses = ["notice"],
)

# Do NOT directly depend on `quantize_model_cc_impl` unless it is necessary
# (i.e. undefined symbol). See the comments in `quantize_model_cc`.
cc_library(
    name = "quantize_model_cc_impl",
    srcs = ["quantize_model.cc"],
    hdrs = ["quantize_model.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        # Directly linked to `libmachina_cc.so` or
        # `_pywrap_machina_internal.so` if static build.
        "//machina:__pkg__",
        "//machina/python:__pkg__",
    ],
    deps = [
        ":py_function_lib",
        "//machina/cc/saved_model:loader",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo/cc:config",
        "//machina/compiler/mlir/quantization/stablehlo/cc:context",
        "//machina/compiler/mlir/quantization/stablehlo/cc:debugger",
        "//machina/compiler/mlir/quantization/stablehlo/cc:io",
        "//machina/compiler/mlir/quantization/stablehlo/cc:post_calibration",
        "//machina/compiler/mlir/quantization/stablehlo/cc:pre_calibration",
        "//machina/compiler/mlir/quantization/stablehlo/cc:saved_model_export",
        "//machina/compiler/mlir/quantization/stablehlo/cc:saved_model_import",
        "//machina/compiler/mlir/quantization/stablehlo/cc:types",
        "//machina/compiler/mlir/quantization/stablehlo/cc:weight_only_ptq",
        "//machina/compiler/mlir/quantization/stablehlo/cc/calibration:component",
        "//machina/compiler/mlir/quantization/stablehlo/cc/calibration:statistics",
        "//machina/compiler/mlir/quantization/machina:exported_model_proto_cc",
        "//machina/compiler/mlir/quantization/machina:passes",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/quantization/machina:quantize_passes",
        "//machina/compiler/mlir/quantization/machina:quantize_preprocess",
        "//machina/compiler/mlir/quantization/machina/calibrator:calibration_statistics_saver_op",  # Required for CalibrationStatisticsSaver op registration.
        "//machina/compiler/mlir/quantization/machina/calibrator:custom_aggregator_op",  # Required for CustomAggregator op registration.
        "//machina/compiler/mlir/quantization/machina/cc:run_passes",
        "//machina/compiler/mlir/quantization/machina/debugging:dump_tensor_op",  # Required for DumpTensor op registration.
        "//machina/compiler/mlir/machina:mlir_import_options",
        "//machina/compiler/mlir/machina:translate_lib",
        "//machina/compiler/mlir/machina/transforms:tf_dialect_passes",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:path",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@local_tsl//tsl/platform:path",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

# OSS: This is a header-only target. The implementation target `quantize_model_cc_impl` is
# directly linked to `lib_pywrap_machina_internal.so`, so in most use cases of python-
# exported symbols depending directly on `quantize_model_cc_impl` should be unnecessary.
# Using the header-only target will help avoid the ODR violation.
cc_library(
    name = "quantize_model_cc",
    hdrs = ["quantize_model.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":py_function_lib",
        "//machina/compiler/mlir/quantization/machina:exported_model_proto_cc",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

pytype_strict_library(
    name = "py_function_lib_py",
    srcs = ["py_function_lib.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_function_lib",
        ":representative_dataset",
        ":save_model",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//machina/compiler/mlir/quantization/machina:exported_model_proto_py",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_py",
        "//machina/compiler/mlir/quantization/machina/calibrator:calibration_algorithm",
        "//machina/compiler/mlir/quantization/machina/calibrator:calibration_statistics_proto_py",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/eager:wrap_function",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/lib/io:file_io",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:loader",
        "//machina/python/trackable:autotrackable",
        "//machina/python/types:core",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "py_function_lib_py_test",
    srcs = ["py_function_lib_test.py"],
    main = "py_function_lib_test.py",
    deps = ["//machina/python/platform:client_testlib"],
)

cc_library(
    name = "type_casters",
    hdrs = ["type_casters.h"],
    copts = ["-fexceptions"],  # Required for pybind11.
    # Required for pybind11.
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],
    deps = [
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/machina:exported_model_proto_cc",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/quantization/machina/calibrator:calibration_statistics_proto_cc",
        "//machina/core:protos_all_cc",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//third_party/python_runtime:headers",  # build_cleaner: keep; Required for pybind11.
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
    ],
)

cc_library(
    name = "py_function_lib",
    hdrs = ["py_function_lib.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo/cc/calibration:min_max_value",
        "//machina/compiler/mlir/quantization/machina:exported_model_proto_cc",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/quantization/machina/calibrator:calibration_statistics_proto_cc",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "unfreeze_constants",
    srcs = ["unfreeze_constants.cc"],
    hdrs = ["unfreeze_constants.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//machina/compiler/mlir/quantization/machina:passes",
        "//machina/compiler/mlir/quantization/machina/cc:run_passes",
        "//machina/compiler/mlir/quantization/machina/cc:save_variables",
        "//machina/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_python_pybind_extension(
    name = "pywrap_function_lib",
    srcs = ["pywrap_function_lib.cc"],
    pytype_srcs = ["pywrap_function_lib.pyi"],
    visibility = [
        "__subpackages__",
        "//machina/lite/python:__subpackages__",
        "//machina/python:__pkg__",
        "//machina/tools/pip_package:__subpackages__",
    ],
    deps = [
        ":py_function_lib",
        ":type_casters",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//machina/compiler/mlir/quantization/stablehlo/cc/calibration:min_max_value",
        "//machina/compiler/mlir/quantization/machina:exported_model_proto_cc",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/compiler/mlir/quantization/machina/calibrator:calibration_statistics_proto_cc",
        "//machina/core:protos_all_cc",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@pybind11",
    ],
)

# Exports python symbols via pybind11.
tf_python_pybind_extension(
    name = "pywrap_quantize_model",
    srcs = ["pywrap_quantize_model.cc"],
    pytype_srcs = ["pywrap_quantize_model.pyi"],
    # All deps must be header-only.
    deps = [
        ":py_function_lib",
        ":type_casters",
        "//machina/compiler/mlir/quantization/machina:exported_model_proto_cc",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/core:protos_all_cc",
        "//machina/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:import_status_module",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ] + if_pywrap(
        [":quantize_model_cc_impl"],
        [":quantize_model_cc"],
    ),
)

tf_py_strict_test(
    name = "pywrap_quantize_model_test",
    srcs = [
        "pywrap_quantize_model_test.py",
    ],
    deps = [
        ":py_function_lib_py",
        ":pywrap_quantize_model",
        "//machina:machina_py",
        "//machina/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "save_model",
    srcs = [
        "save_model.py",
    ],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/lib/io:file_io",
        "//machina/python/saved_model:builder",
        "//machina/python/saved_model:constants",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:tag_constants",
        "//machina/python/training:saver",
        "@absl_py//absl/logging",
    ],
)

pytype_strict_library(
    name = "quantize_model",
    srcs = [
        "quantize_model.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":py_function_lib_py",
        ":pywrap_quantize_model",
        ":representative_dataset",
        ":save_model",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_py",
        "//machina/core:protos_all_py",
        "//machina/python/lib/io:file_io",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:tag_constants",
        "//machina/python/trackable:autotrackable",
        "//machina/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

tf_py_strict_test(
    name = "quantize_model_test",
    size = "medium",
    srcs = ["integration_test/quantize_model_test.py"],
    shard_count = 50,  # Parallelize the test to avoid timeouts.
    tags = [
        "no_mac",  # TODO(b/292100835): Reenable
    ],
    deps = [
        ":quantize_model",
        ":quantize_model_test_base",
        ":representative_dataset",
        ":save_model",
        "//machina:machina_py",
        "//machina/compiler/mlir/quantization/common/python:testing",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_py",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/lib/io:tf_record",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:builder",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:save",
        "//machina/python/saved_model:save_options",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:signature_def_utils",
        "//machina/python/saved_model:tag_constants",
        "//machina/python/training:checkpoint_utils",
        "//machina/python/types:core",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "quantize_model_test_base",
    testonly = 1,
    srcs = ["integration_test/quantize_model_test_base.py"],
    tags = ["no_pip"],
    deps = [
        ":representative_dataset",
        "//machina:machina_py",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/lib/io:file_io",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/ops/ragged:ragged_string_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:builder",
        "//machina/python/saved_model:save",
        "//machina/python/saved_model:signature_def_utils",
        "//machina/python/trackable:asset",
        "//machina/python/trackable:autotrackable",
        "//machina/python/types:core",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "concurrency_test",
    size = "medium",
    srcs = ["integration_test/concurrency_test.py"],
    deps = [
        ":quantize_model",
        "//machina:machina_py",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_py",
        "//machina/python/eager:def_function",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:save",
        "//machina/python/saved_model:tag_constants",
        "//machina/python/trackable:autotrackable",
    ],
)

pytype_strict_library(
    name = "representative_dataset",
    srcs = [
        "representative_dataset.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_py",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/data/ops:readers",
        "//machina/python/eager:context",
        "//machina/python/framework:tensor_util",
        "//machina/python/lib/io:python_io",
        "//machina/python/platform:tf_logging",
        "//machina/python/types:core",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "representative_dataset_test",
    srcs = ["representative_dataset_test.py"],
    deps = [
        ":representative_dataset",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//machina/python/types:core",
        "//third_party/py/numpy",
    ],
)
