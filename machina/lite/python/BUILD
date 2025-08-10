load("@flatbuffers//:build_defs.bzl", "flatbuffer_py_library")
load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//machina:pytype.default.bzl", "pytype_strict_contrib_test", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("//machina:machina.default.bzl", "get_compatible_with_portable", "pywrap_binaries", "pywrap_library")
load("//machina/core/platform:build_config_root.bzl", "if_pywrap")
load("//machina/lite:special_rules.bzl", "internal_visibility_allowlist")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:__subpackages__",
        "//machina:internal",
        "//third_party/odml/infra/genai/conversion:__subpackages__",
        "//third_party/odml/model_customization/quantization:__subpackages__",
        "//third_party/py/ai_edge_torch:__subpackages__",
        "//third_party/py/machina_federated:__subpackages__",
        "//third_party/tflite_micro:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files([
    "tflite_convert.py",
    "pywrap_tflite_common.json",
    "pywrap_tflite_common.lds",
    "pywrap_tflite_common_darwin.lds",
])

flatbuffer_py_library(
    name = "schema_py",
    srcs = ["//machina/compiler/mlir/lite/schema:schema.fbs"],
)

flatbuffer_py_library(
    name = "conversion_metadata_schema_py",
    srcs = ["//machina/compiler/mlir/lite/schema:conversion_metadata.fbs"],
)

py_strict_library(
    name = "interpreter",
    srcs = [
        "interpreter.py",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//machina/lite/python/interpreter_wrapper:_pywrap_machina_interpreter_wrapper",
        "//machina/lite/python/metrics",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "interpreter_test",
    srcs = ["interpreter_test.py"],
    data = [
        "//machina/lite:testdata/sparse_tensor.bin",
        "//machina/lite/python/testdata:interpreter_test_data",
        "//machina/lite/python/testdata:test_delegate.so",
    ],
    # Static linking is required because this loads a cc_binary as a shared
    # library, which would otherwise create ODR violations.
    # copybara:uncomment linking_mode = "static",
    tags = [
        "no_oss",  # TODO(b/190842754): Enable test in OSS.
    ],
    deps = [
        ":interpreter",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina:machina_py",
        "//machina/lite/python:lite",
        "//machina/lite/python/metrics",
        "//machina/lite/python/testdata:_pywrap_test_registerer",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
    ],
)

py_strict_binary(
    name = "tflite_convert",
    srcs = ["tflite_convert.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_convert_main_lib",
        "//machina:machina_py",
        "//machina/lite/python:lite",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "tflite_convert_main_lib",
    srcs = ["tflite_convert.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":convert",
        "//machina:machina_py",
        "//machina/lite/toco:toco_flags_proto_py",
        "//machina/lite/toco/logging:gen_html",
        "//machina/python:tf2",
        "//machina/python/framework:dtypes",
        "//machina/python/platform:gfile",
        "//machina/python/util:keras_deps",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "test_util",
    srcs = ["test_util.py"],
    deps = [
        ":lite",
        ":schema_py",
        ":schema_util",
        "//machina/lite/tools:visualize_lib",
    ],
)

py_strict_test(
    name = "test_util_test",
    srcs = ["test_util_test.py"],
    data = [
        "//machina/lite:testdata/add.bin",
        "//machina/lite:testdata/softplus_flex.bin",
    ],
    deps = [
        ":test_util",
        #internal proto upb dep
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/platform:resource_loader",
    ],
)

py_strict_test(
    name = "tflite_convert_test",
    srcs = ["tflite_convert_test.py"],
    data = [
        ":tflite_convert.par",
        "@tflite_mobilenet_ssd_quant_protobuf//:tflite_graph.pb",
    ],
    # Increased thread count for reducing timeout failures.
    shard_count = 10,
    tags = [
        "no_oss",
        "no_windows",
    ],
    deps = [
        ":convert",
        ":test_util",
        ":tflite_convert_main_lib",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina:machina_py",
        "//machina/core:protos_all_py",
        "//machina/python:tf2",
        "//machina/python/client:session",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/platform:resource_loader",
        "//machina/python/saved_model",
        "//machina/python/saved_model:save",
        "//machina/python/trackable:autotrackable",
        "//machina/python/training:training_util",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "lite",
    srcs = ["lite.py"],
    tags = [
        "ignore_for_dep=third_party.py.keras",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":conversion_metadata_schema_py",
        ":convert",
        ":convert_phase",
        ":convert_saved_model",
        ":interpreter",
        ":lite_constants",
        ":op_hint",
        ":util",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//machina/compiler/mlir/quantization/machina/python:representative_dataset",
        "//machina/core:protos_all_py",
        "//machina/lite/experimental/microfrontend:audio_microfrontend_py",
        "//machina/lite/profiling/proto:model_runtime_info_py",
        "//machina/lite/profiling/proto:profiling_info_py",
        "//machina/lite/python/metrics",
        "//machina/lite/python/optimize:calibrator",
        "//machina/lite/tools:flatbuffer_utils",
        "//machina/lite/tools/optimize/debugging/python:debugger",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:function",
        "//machina/python/framework:byte_swap_tensor",
        "//machina/python/framework:convert_to_constants",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:versions",
        "//machina/python/platform:gfile",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:save",
        "//machina/python/saved_model:save_options",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:tag_constants",
        "//machina/python/util:deprecation",
        "//machina/python/util:keras_deps",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "@absl_py//absl/logging",
    ],
)

py_strict_test(
    name = "lite_test",
    srcs = ["lite_test.py"],
    data = [
        "//machina/lite/python/testdata:control_flow_v1.pbtxt",
        "@tflite_mobilenet_ssd_quant_protobuf//:tflite_graph.pb",
    ],
    shard_count = 4,
    tags = [
        "no_windows",
    ],
    deps = [
        ":conversion_metadata_schema_py",
        ":convert",
        ":interpreter",
        ":lite",
        ":lite_constants",
        ":schema_py",
        ":util",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina:machina_py",
        "//machina/python/client:session",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:convert_to_constants",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:versions",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:logging_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/platform:resource_loader",
        "//machina/python/saved_model",
        "//machina/python/training:training_util",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "lite_v2_test",
    srcs = ["lite_v2_test.py"],
    data = [
        "//machina/lite/python/testdata:test_delegate.so",
        "//machina/lite/python/testdata/control_flow_v1_saved_model:saved_model.pb",
    ],
    shard_count = 18,
    tags = [
        "no_windows",
    ],
    deps = [
        ":conversion_metadata_schema_py",
        ":convert",
        ":interpreter",
        ":lite",
        ":lite_v2_test_util",
        ":schema_py",
        ":test_util",
        ":util",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina:machina_py",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_options_proto_py",
        "//machina/lite/python/testdata:_pywrap_test_registerer",
        "//machina/lite/python/testdata:double_op",
        "//machina/lite/tools:flatbuffer_utils",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:versions",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:map_ops",
        "//machina/python/ops:rnn",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
        "//machina/python/saved_model",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:save",
        "//machina/python/saved_model:save_options",
        "//machina/python/trackable:autotrackable",
        "@absl_py//absl/testing:parameterized",
        "@pypi_jax//:pkg",
    ],
)

py_strict_library(
    name = "lite_v2_test_util",
    testonly = 1,
    srcs = ["lite_v2_test_util.py"],
    tags = [
        "no_windows",
    ],
    deps = [
        ":interpreter",
        ":lite",
        "//machina:machina_py_no_contrib",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/trackable:autotrackable",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "lite_flex_test",
    srcs = ["lite_flex_test.py"],
    deps = [
        ":convert",
        ":interpreter",
        ":lite",
        ":test_util",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina:machina_py",
        "//machina/core:protos_all_py",
        "//machina/lite/python/testdata:double_op",
        "//machina/python/client:session",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model",
        "//machina/python/trackable:autotrackable",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "util",
    srcs = ["util.py"],
    visibility = internal_visibility_allowlist(),
    deps = [
        ":conversion_metadata_schema_py",
        ":op_hint",
        ":schema_py",
        ":schema_util",
        ":tflite_keras_util",
        "//machina/core:protos_all_py",
        "//machina/lite/tools:flatbuffer_utils",
        "//machina/python/eager:function",
        "//machina/python/framework:convert_to_constants",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:error_interpolation",
        "//machina/python/grappler:tf_optimizer",
        "//machina/python/training:saver",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
        "@flatbuffers//:runtime_py",
    ],
)

py_strict_test(
    name = "util_test",
    srcs = ["util_test.py"],
    tags = [
        "no_windows",
    ],
    deps = [
        ":util",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina:machina_py",
        "//machina/lite/python:lite",
        "//machina/lite/tools:flatbuffer_utils",
        "//machina/python/client:session",
        "//machina/python/framework:convert_to_constants",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "tflite_keras_util",
    srcs = [
        "tflite_keras_util.py",
    ],
    deps = [
        "//machina/python/eager:def_function",
        "//machina/python/framework:tensor_spec",
        "//machina/python/util:compat",
        "//machina/python/util:keras_deps",
        "//machina/python/util:nest",
    ],
)

py_strict_library(
    name = "lite_constants",
    srcs = ["lite_constants.py"],
    deps = [
        "//machina/compiler/mlir/lite:converter_flags_proto_py",
        "//machina/python/framework:dtypes",
        "//machina/python/util:all_util",
        "//machina/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "convert",
    srcs = ["convert.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":convert_phase",
        ":lite_constants",
        ":util",
        "//machina/compiler/mlir/lite:converter_flags_proto_py",
        "//machina/compiler/mlir/lite:model_flags_proto_py",
        "//machina/compiler/mlir/lite:types_proto_py",
        "//machina/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "//machina/compiler/mlir/lite/python:wrap_converter",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_config_proto_py",
        "//machina/compiler/mlir/quantization/stablehlo:quantization_options_proto_py",
        "//machina/lite/python/metrics:metrics_wrapper",
        "//machina/lite/tools:flatbuffer_utils",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_shape",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ] + if_pywrap(
        if_true = [":pywrap_tflite"],
    ),
)

py_strict_library(
    name = "op_hint",
    srcs = ["op_hint.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:graph_util",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/util:all_util",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

pytype_strict_contrib_test(
    name = "convert_test",
    srcs = ["convert_test.py"],
    deps = [
        ":convert",
        ":interpreter",
        ":op_hint",
        "//machina/compiler/mlir/lite:converter_flags_proto_py",
        "//machina/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "//machina/lite/python/metrics:metrics_wrapper",
        "//machina/python/client:session",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:graph_util",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ] + if_pywrap(
        if_true = [":pywrap_tflite"],
    ),
)

py_strict_library(
    name = "convert_saved_model",
    srcs = ["convert_saved_model.py"],
    visibility = [
        "//machina/lite:__subpackages__",
    ],
    deps = [
        ":convert_phase",
        ":util",
        "//machina/python/client:session",
        "//machina/python/framework:graph_util",
        "//machina/python/framework:ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model",
        "//machina/python/saved_model:constants",
        "//machina/python/saved_model:loader",
    ],
)

py_strict_test(
    name = "convert_saved_model_test",
    srcs = ["convert_saved_model_test.py"],
    tags = [
        "no_windows",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":convert_saved_model",
        #internal proto upb dep
        "//machina/python/client:session",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/layers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops/losses",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//machina/python/saved_model",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:tag_constants",
    ],
)

py_strict_binary(
    name = "convert_file_to_c_source",
    srcs = ["convert_file_to_c_source.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":util",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

sh_test(
    name = "convert_file_to_c_source_test",
    srcs = ["convert_file_to_c_source_test.sh"],
    data = [":convert_file_to_c_source"],
)

py_strict_library(
    name = "schema_util",
    srcs = ["schema_util.py"],
    visibility = ["//machina/lite/schema:utils_friends"],
    deps = [
        "//machina/python/util:all_util",
    ],
)

# Use py_library since the metrics module is imported in a try-except block,
# which doesn't work with the pytype_strict_library.
py_strict_library(
    name = "convert_phase",
    srcs = ["convert_phase.py"],
    visibility = ["//machina/lite:__subpackages__"],
    deps = [
        "//machina/compiler/mlir/lite/metrics:converter_error_data_proto_py",
        "//machina/lite/python/metrics",
    ],
)

py_strict_library(
    name = "analyzer",
    srcs = [
        "analyzer.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/lite/python:wrap_converter",
        "//machina/python/util:tf_export",
    ] + if_pywrap(
        if_false = [
            "//machina/lite/python/analyzer_wrapper:_pywrap_analyzer_wrapper",
        ],
        if_true = [
            ":pywrap_tflite",
        ],
    ),
)

py_strict_test(
    name = "analyzer_test",
    srcs = ["analyzer_test.py"],
    data = [
        "//machina/lite:testdata/add.bin",
        "//machina/lite:testdata/conv_huge_im2col.bin",
        "//machina/lite:testdata/multi_add_flex.bin",
    ],
    deps = [
        ":analyzer",
        #internal proto upb dep
        "//machina:machina_py",
        "//machina/lite/python:lite",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
        "//machina/python/trackable:autotrackable",
    ],
)

# Use pywrap_library to avoid duplicate registration of pybind11 modules.
# A great example on how to use pywrap_library is
# https://github.com/vam-google/symbol-locations/blob/main/pybind/BUILD
# The following pywrap_library is used by LiteRT repo to avoid shared links provided
# by Tensorflow under machina/python:_pywrap_machina
# This isolate LiteRT's pybind11 dependencies. To use, add pybind deps under pywrap_tflite
# and refer pywrap_tflite to any target that needsd to selected shared objects.
py_strict_library(
    name = "tflite_pywrap_deps",
    visibility = ["//visibility:public"],
    deps = [
        "//machina/lite/experimental/genai:pywrap_genai_ops",
        "//machina/lite/python/analyzer_wrapper:_pywrap_analyzer_wrapper",
        "//machina/lite/python/interpreter_wrapper:_pywrap_machina_interpreter_wrapper",
        "//machina/lite/python/metrics:_pywrap_machina_lite_metrics_wrapper",
        "//machina/lite/python/optimize:_pywrap_machina_lite_calibration_wrapper",
        "//machina/lite/testing:_pywrap_string_util",
        "//machina/lite/tools/optimize/python:_pywrap_modify_model_interface",
        "//machina/lite/tools/optimize/sparsity:format_converter_wrapper_pybind11",
    ],
)

pywrap_library(
    name = "pywrap_tflite",
    common_lib_def_files_or_filters = {
        "machina/lite/python/pywrap_tflite_common": "pywrap_tflite_common.json",
    },
    common_lib_version_scripts = {
        "machina/lite/python/pywrap_tflite_common": select({
            "@bazel_tools//src/conditions:windows": None,
            "@bazel_tools//src/conditions:darwin": "pywrap_tflite_common_darwin.lds",
            "//conditions:default": "pywrap_tflite_common.lds",
        }),
    },
    pywrap_count = 8,
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_pywrap_deps",
    ],
)

pywrap_binaries(
    name = "pywrap_tflite_binaries",
    dep = ":pywrap_tflite",
)
