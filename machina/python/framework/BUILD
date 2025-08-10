# python/framework package

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")

# Placeholder: load py_proto_library
load(
    "//machina:machina.bzl",
    "if_cuda_tools",
    "if_not_windows",
    "if_oss",
    "if_xla_available",
    "tf_cc_binary",
    "tf_cc_shared_object",
    "tf_cc_test",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)
load("//machina:machina.default.bzl", "cuda_py_strict_test", "stripped_cc_info", "tf_py_strict_test", "tf_python_framework_friends", "tf_python_pybind_extension")
load("//machina/core/platform:build_config.bzl", "pyx_library", "tf_additional_all_protos", "tf_additional_lib_deps", "tf_proto_library", "tf_protos_grappler")  # @unused
load("//machina/core/platform:build_config_root.bzl", "if_pywrap", "if_static", "tf_additional_xla_deps_py")
load("//machina/python/tpu:tpu.bzl", "tpu_py_strict_test")
load(
    "//machina/tools/test:performance.bzl",
    "cuda_py_benchmark_test",
)

visibility = tf_python_framework_friends()  # buildifier: disable=package-on-top

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

cc_library(
    name = "test_file_system",
    srcs = ["test_file_system.cc"],
    copts = if_not_windows(["-Wno-sign-compare"]),
    linkopts = select({
        "//conditions:default": [
            "-lm",
        ],
        "//machina:macos": [],
        "//machina:windows": [],
    }),
    deps = [
        "//machina/core:lib",
        "//machina/core/platform:null_file_system",
    ],
)

stripped_cc_info(
    name = "test_file_system_stripped",
    deps = [":test_file_system"],
)

tf_cc_shared_object(
    name = "test_file_system.so",
    srcs = if_pywrap(if_false = ["test_file_system.cc"]),
    copts = if_not_windows(["-Wno-sign-compare"]),
    deps = if_pywrap(
        if_false = [
            "@eigen_archive//:eigen3",
            "//machina/core:framework_headers_lib",
        ],
        if_true = [
            ":test_file_system_stripped",
            "//machina/python:machina_common_framework",
        ],
    ) + ["@com_google_protobuf//:protobuf_headers"],
)

tf_py_strict_test(
    name = "file_system_test",
    size = "small",
    srcs = ["file_system_test.py"],
    data = [":test_file_system.so"],
    main = "file_system_test.py",
    tags = [
        "no_pip",  # Path issues due to test environment
        "no_windows",
        "notap",
    ],
    deps = [
        ":dtypes",
        ":load_library",
        ":test_lib",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:io_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
        "//machina/python/util:compat",
    ],
)

cc_library(
    name = "python_op_gen",
    srcs = [
        "python_op_gen.cc",
    ],
    hdrs = [
        "python_op_gen.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":op_reg_offset_proto_cc",
        ":python_op_gen_annotator",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:op_gen_lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:numbers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:protobuf",
    ],
    alwayslink = 1,
)

cc_library(
    name = "python_op_gen_annotator",
    srcs = ["python_op_gen_annotator.cc"],
    hdrs = ["python_op_gen_annotator.h"],
    visibility = ["//machina:internal"],
    deps = [
        ":kythe_metadata_proto_cc",
        ":op_reg_offset_proto_cc",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "python_op_gen_annotator_test",
    srcs = ["python_op_gen_annotator_test.cc"],
    deps = [
        ":kythe_metadata_proto_cc",
        ":python_op_gen_annotator",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:protobuf",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_python_op_gen",
    srcs = ["python_op_gen_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_python_op_gen.pyi",
    ],
    deps = [
        "//machina/python:python_op_gen_headers_lib",
        "//machina/python/lib/core:pybind11_absl",
        "//machina/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

cc_library(
    name = "python_op_gen_main",
    srcs = ["python_op_gen_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":op_reg_offset_proto_cc",
        ":python_op_gen",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:op_gen_lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "python_op_gen_test",
    srcs = ["python_op_gen_test.cc"],
    deps = [
        ":kythe_metadata_proto_cc",
        ":op_reg_offset_proto_cc",
        ":python_op_gen",
        "//machina/core:framework",
        "//machina/core:op_gen_lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

tf_proto_library(
    name = "op_reg_offset_proto",
    srcs = ["op_reg_offset.proto"],
    visibility = ["//machina:internal"],
)

tf_proto_library(
    name = "kythe_metadata_proto",
    srcs = ["kythe_metadata.proto"],
    visibility = ["//machina:internal"],
)

tf_cc_binary(
    name = "offset_counter",
    srcs = ["offset_counter.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":offset_counter_helper",
        ":op_reg_offset_proto_cc",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:strcat",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:types",
        "@local_xla//xla/tsl/util:command_line_flags",
    ],
)

cc_library(
    name = "offset_counter_helper",
    srcs = ["offset_counter_helper.cc"],
    hdrs = ["offset_counter_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":op_reg_offset_proto_cc",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:regexp",
        "@local_tsl//tsl/platform:strcat",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:types",
    ],
)

tf_cc_test(
    name = "offset_counter_helper_test",
    srcs = ["offset_counter_helper_test.cc"],
    deps = [
        ":offset_counter_helper",
        ":op_reg_offset_proto_cc",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/strings:str_format",
    ],
)

# Do not depend on this rule! Depend on the fine-grained sub-targets instead.
py_strict_library(
    name = "for_generated_wrappers",
    tags = ["avoid_dep"],
    visibility = ["//visibility:public"],
    deps = [
        ":byte_swap_tensor",
        ":device",
        ":device_spec",
        ":dtypes",
        ":function",
        ":op_def_library",
        ":op_def_registry",
        ":ops",
        ":registry",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":versions",
    ],
)

# This rule should only be depended on by tf_gen_op_wrapper_py.
# Do not depend on this rule! Depend on the fine-grained sub-targets instead.
# This is the same as "for_generated_wrappers" minus the "function" dep.
# This is to avoid circular dependencies, as "function" uses generated op wrappers.
py_strict_library(
    name = "for_generated_wrappers_v2",
    tags = ["avoid_dep"],
    visibility = ["//visibility:public"],
    deps = [
        ":byte_swap_tensor",
        ":constant_op",
        ":device",
        ":device_spec",
        ":dtypes",
        ":op_def_library",
        ":op_def_registry",
        ":ops",
        ":registry",
        ":tensor_shape",
        ":versions",
        "//machina/python/eager:context",
        "//machina/python/eager:core",
        "//machina/python/eager:execute",
        "//machina/security/fuzzing/py:annotation_types",
        "//machina/tools/docs:doc_controls",
        "@pypi_typing_extensions//:pkg",
    ],
)

py_strict_library(
    name = "subscribe",
    srcs = ["subscribe.py"],
    deps = [
        ":ops",
        ":tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "byte_swap_tensor",
    srcs = ["byte_swap_tensor.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":dtypes",
        "//machina/core:protos_all_py",
    ],
)

py_strict_library(
    name = "c_api_util",
    srcs = ["c_api_util.py"],
    visibility = visibility + ["//machina:internal"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/util:compat",
        "//machina/python/util:tf_contextlib",
    ],
)

py_strict_library(
    name = "common_shapes",
    srcs = ["common_shapes.py"],
    deps = [
        ":tensor_shape",
    ],
)

py_strict_library(
    name = "constant_op",
    srcs = [
        "constant_op.py",
        "constant_tensor_conversion.py",
    ],
    visibility = visibility + [
        "//smartass:__subpackages__",
        "//machina:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":dtypes",
        ":ops",
        ":tensor",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tensor_util",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/eager:execute",
        "//machina/python/profiler:trace",
        "//machina/python/saved_model:nested_structure_coder",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "device_spec",
    srcs = ["device_spec.py"],
    visibility = visibility + ["//third_party/py/tf_slim:__subpackages__"],
    deps = [
        "//machina/python:pywrap_tfe",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "device",
    srcs = ["device.py"],
    visibility = visibility + ["//machina:internal"],
    deps = [
        ":device_spec",
        "//machina/python:tf2",
    ],
)

tf_python_pybind_extension(
    name = "_dtypes",
    srcs = ["dtypes.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_dtypes.pyi",
    ],
    visibility = visibility + ["//third_party/py/tensorfn:__subpackages__"],
    deps = [
        "//machina/core:framework_headers_lib",
        "//machina/core:protos_all_cc",
        "@eigen_archive//:eigen3",
        "@pybind11",
    ],
)

py_strict_library(
    name = "dtypes",
    srcs = ["dtypes.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//ads/adsml/ab:atls_configs",
    # "//smartass:__subpackages__",
    # "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    # "//third_party/mlperf:__subpackages__",
    # "//third_party/py/tf_slim:__subpackages__",
    # "//machina:internal",
    # "//machina_models:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":_dtypes",
        # copybara:uncomment_begin(google-only)
        # ":cpp_shape_inference_proto_py",
        # "//third_party/py/numpy",
        # copybara:uncomment_end
        "//machina/core:protos_all_py",
        "//machina/core/function/trace_type",
        "//machina/python/types:trace",
        "//machina/python:pywrap_machina",
        "//machina/python/types:doc_typealias",
        "//machina/python/util:tf_export",
        "//machina/tools/docs:doc_controls",
        "@ml_dtypes_py//ml_dtypes",
    ],
)

py_strict_library(
    name = "errors",
    srcs = [
        "errors.py",
        "errors_impl.py",
    ],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//smartass/brain/ops:__subpackages__",
    # "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    # "//third_party/py/tf_slim:__subpackages__",
    # "//machina:internal",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":c_api_util",
        "//machina/core:protos_all_py",
        "//machina/python:_pywrap_py_exception_registry",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
    ],
)

py_strict_library(
    name = "error_interpolation",
    srcs = [
        "error_interpolation.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/util:tf_stack",
    ],
)

tf_cc_binary(
    name = "_native_proto_caster",
    linkshared = True,
    deps = ["@pybind11_protobuf//pybind11_protobuf:native_proto_caster"],
)

py_strict_library(
    name = "function",
    srcs = ["function.py"],
    data = [":_native_proto_caster"],  # copybara:comment
    visibility = visibility + [
        "//smartass/brain:__subpackages__",
        "//machina:internal",
        "//machina_models:__subpackages__",
        "//third_party/cloud_tpu/convergence_tools:__subpackages__",
        "//third_party/mlperf:__subpackages__",
    ],
    deps = [
        ":c_api_util",
        ":dtypes",
        ":graph_to_function_def",
        ":ops",
        ":tensor",
        "//machina/core:protos_all_py",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/eager:context",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/util:compat",
        "//machina/python/util:function_utils",
        "//machina/python/util:tf_contextlib",
        "//machina/python/util:tf_inspect",
        # copybara:uncomment "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

py_strict_library(
    name = "graph_to_function_def",
    srcs = ["graph_to_function_def.py"],
    deps = [
        ":op_def_registry",
        "//machina/core:protos_all_py",
    ],
)

py_strict_library(
    name = "function_def_to_graph",
    srcs = ["function_def_to_graph.py"],
    visibility = visibility,
    deps = [
        ":cpp_shape_inference_proto_py",
        ":func_graph",
        ":importer",
        ":ops",
        ":versions",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/ops:resource_variable_ops",
    ],
)

pytype_strict_library(
    name = "strict_mode",
    srcs = ["strict_mode.py"],
    deps = [
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "function_def_to_graph_test",
    size = "small",
    srcs = ["function_def_to_graph_test.py"],
    tags = ["no_pip"],
    deps = [
        ":constant_op",
        ":dtypes",
        ":function_def_to_graph",
        ":graph_to_function_def",
        ":op_def_library",
        ":ops",
        ":tensor_shape",
        ":tensor_spec",
        ":test_lib",
        ":test_ops",
        "//machina/python/eager:def_function",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "graph_util",
    srcs = [
        "graph_util.py",
        "graph_util_impl.py",
    ],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + ["//machina:internal"],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":_proto_comparators",
        ":dtypes",
        ":ops",
        "//machina/core:protos_all_py",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "convert_to_constants",
    srcs = [
        "convert_to_constants.py",
    ],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//learning/deepmind/deepfunc:__subpackages__",
    # "//platforms/darwinn:__subpackages__",
    # "//machina:internal",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":dtypes",
        ":errors",
        ":graph_util",
        ":ops",
        ":tensor_util",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/eager:wrap_function",
        "//machina/python/grappler:tf_optimizer",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:saver",
        "//machina/python/util:deprecation",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "kernels",
    srcs = [
        "kernels.py",
    ],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/util:compat",
    ],
)

py_strict_library(
    name = "op_def_library",
    srcs = ["op_def_library.py"],
    deps = [
        ":dtypes",
        ":op_callbacks",
        ":op_def_library_pybind",
        ":op_def_registry",
        ":ops",
        ":tensor",
        ":tensor_shape",
        "//machina/core:protos_all_py",
        "//machina/core/config:flags_py",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:_pywrap_utils",
        "//machina/python/util:compat",
        "//machina/python/util:tf_contextlib",
    ],
)

tf_python_pybind_extension(
    name = "_op_def_library_pybind",
    srcs = ["op_def_library_pybind.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_op_def_library_pybind.pyi",
    ],
    deps = if_pywrap(
        if_false = [":op_def_util_headers"],
        if_true = [":op_def_util_cc"],
    ) + [
        "//machina/core:framework_headers_lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/python/lib/core:pybind11_status",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

pytype_strict_library(
    name = "op_def_library_pybind",
    srcs = ["op_def_library_pybind.py"],
    deps = [
        ":_op_def_library_pybind",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_machina",
    ],
)

tf_python_pybind_extension(
    name = "_op_def_registry",
    srcs = ["op_def_registry.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_op_def_registry.pyi",
    ],
    deps = [
        "//machina/core:framework_headers_lib",
        "//machina/core:protos_all_cc",
        "//machina/python/lib/core:pybind11_status",
        "@pybind11",
    ],
)

py_strict_library(
    name = "op_def_registry",
    srcs = ["op_def_registry.py"],
    visibility = visibility + ["//machina:internal"],
    deps = [
        ":_op_def_registry",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_machina",
    ],
)

py_strict_library(
    name = "override_binary_operator",
    srcs = ["override_binary_operator.py"],
    deps = [
        ":dtypes",
        ":ops",
        ":tensor",
        ":tensor_shape",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops/numpy_ops:np_dtypes",
        "//machina/python/util:nest",
        "//machina/python/util:traceback_utils",
        "//third_party/py/numpy",
    ],
)

cc_library(
    name = "py_context_manager",
    srcs = ["py_context_manager.cc"],
    hdrs = ["py_context_manager.h"],
    deps = [
        "//machina/core:lib",  # for core/platform/logging.h
        "//machina/python/lib/core:safe_pyobject_ptr",
        "@local_xla//third_party/python_runtime:headers",
    ],
)

# Pybind extension used by py_context_manager_test.
tf_python_pybind_extension(
    name = "_py_context_manager",
    srcs = ["py_context_manager_pybind.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_py_context_manager.pyi",
    ],
    starlark_only = True,
    deps = [
        ":py_context_manager",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_py_strict_test(
    name = "py_context_manager_test",
    srcs = ["py_context_manager_test.py"],
    tags = ["no_pip"],
    deps = [
        ":_py_context_manager",
        ":test_lib",
        "//machina/python/platform:test",
    ],
)

cc_library(
    name = "op_def_util_headers",
    textual_hdrs = ["op_def_util.h"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/core:protos_all_cc",
        "//machina/python/lib/core:safe_pyobject_ptr",
    ],
)

cc_library(
    name = "op_def_util_cc",
    srcs = ["op_def_util.cc"],
    hdrs = ["op_def_util.h"],
    deps = [
        "//machina/core:protos_all_cc",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "//machina/python/util:cpp_python_util",
        "@com_google_absl//absl/strings",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

tf_python_pybind_extension(
    name = "_op_def_util",
    srcs = ["op_def_util_pybind.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_op_def_util.pyi",
    ],
    starlark_only = True,
    deps = if_pywrap(
        if_false = [":op_def_util_headers"],
        if_true = [":op_def_util_cc"],
    ) + [
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:protos_all_cc",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_py_strict_test(
    name = "op_def_util_test",
    srcs = ["op_def_util_test.py"],
    tags = [
        "no_pip",
        "no_windows",  # TODO(b/184424727): Enable this test on Windows.
    ],
    deps = [
        ":_op_def_util",
        ":dtypes",
        ":tensor_shape",
        ":test_lib",
        "//machina/core:protos_all_py",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

cc_library(
    name = "python_api_parameter_converter",
    srcs = ["python_api_parameter_converter.cc"],
    hdrs = ["python_api_parameter_converter.h"],
    deps = [
        ":op_def_util_cc",
        ":python_api_info",
        ":python_tensor_converter",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:status",
        "//machina/python/eager:pywrap_tfe_lib",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "//machina/python/util:cpp_python_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

# Note: this target is only used by python_api_parameter_converter_test.
tf_python_pybind_extension(
    name = "_pywrap_python_api_parameter_converter",
    srcs = ["python_api_parameter_converter_wrapper.cc"],
    hdrs = [
        "//machina/c:headers",
        "//machina/c/eager:pywrap_required_hdrs",
        "//machina/c/experimental/ops:pywrap_required_hdrs",
        "//machina/core/common_runtime/eager:pywrap_required_hdrs",
        "//machina/core/distributed_runtime:pywrap_required_hdrs",
        "//machina/core/distributed_runtime/coordination:pywrap_required_hdrs",
        "//machina/core/distributed_runtime/eager:pywrap_required_hdrs",
        "//machina/python/eager:pywrap_required_hdrs",
        "//machina/python/lib/core:safe_pyobject_ptr_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime:pywrap_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime/coordination:pywrap_required_hdrs",
        "@local_xla//xla/tsl/python/lib/core:numpy_hdr",
    ] + if_pywrap(
        if_false = [
            "op_def_util.h",
            "python_api_info.h",
            "python_api_parameter_converter.h",
            "python_tensor_converter.h",
        ],
    ),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_python_api_parameter_converter.pyi",
    ],
    starlark_only = True,
    deps = [
        "//machina/c:pywrap_required_hdrs",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_headers_lib",
        "//machina/core/config:flags_headers",
        "//third_party/py/numpy:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ] + if_pywrap(
        if_true = [
            ":python_api_parameter_converter",
            ":python_api_info",
            ":op_def_util_cc",
            ":python_tensor_converter",
        ],
    ) + if_static(
        extra_deps = [
            "//machina/core/protobuf:eager_service_proto_cc",
            "//machina/core/protobuf:master_proto_cc",
            "//machina/core/protobuf:worker_proto_cc",
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc",
        ],
        otherwise = [
            "//machina/core/protobuf:eager_service_proto_cc_headers_only",
            "//machina/core/protobuf:master_proto_cc_headers_only",
            "//machina/core/protobuf:worker_proto_cc_headers_only",
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc_headers_only",
        ],
    ),
)

tf_py_strict_test(
    name = "python_api_parameter_converter_test",
    srcs = ["python_api_parameter_converter_test.py"],
    tags = ["no_pip"],
    deps = [
        ":_pywrap_python_api_info",
        ":_pywrap_python_api_parameter_converter",
        ":_pywrap_python_tensor_converter",
        ":constant_op",
        ":dtypes",
        ":tensor",
        ":tensor_shape",
        ":test_lib",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/platform:test",
        "//machina/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

cc_library(
    name = "python_api_info",
    srcs = ["python_api_info.cc"],
    hdrs = ["python_api_info.h"],
    deps = [
        ":op_def_util_cc",
        ":python_tensor_converter",
        "//machina/core:framework",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:status",
        "//machina/python/eager:pywrap_tfe_lib",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "//machina/python/util:cpp_python_util",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

# Note: this target is only used by python_api_info_test.
tf_python_pybind_extension(
    name = "_pywrap_python_api_info",
    srcs = ["python_api_info_wrapper.cc"],
    hdrs = [
        "//machina/c:headers",
        "//machina/c/eager:pywrap_required_hdrs",
        "//machina/c/experimental/ops:pywrap_required_hdrs",
        "//machina/core/common_runtime/eager:pywrap_required_hdrs",
        "//machina/core/distributed_runtime:pywrap_required_hdrs",
        "//machina/core/distributed_runtime/coordination:pywrap_required_hdrs",
        "//machina/core/distributed_runtime/eager:pywrap_required_hdrs",
        "//machina/python/eager:pywrap_required_hdrs",
        "//machina/python/lib/core:safe_pyobject_ptr_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime:pywrap_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime/coordination:pywrap_required_hdrs",
        "@local_xla//xla/tsl/python/lib/core:numpy_hdr",
    ] + if_pywrap(
        if_false = [
            "op_def_util.h",
            "python_api_info.h",
            "python_tensor_converter.h",
        ],
    ),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_python_api_info.pyi",
    ],
    starlark_only = True,
    deps = [
        "//machina/c:pywrap_required_hdrs",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_headers_lib",
        "//machina/core/config:flags_headers",
        "//third_party/py/numpy:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ] + if_pywrap(
        if_true = [
            ":python_api_info",
            ":op_def_util_cc",
            ":python_tensor_converter",
        ],
    ) + if_static(
        extra_deps = [
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc",
            "//machina/core/protobuf:eager_service_proto_cc",
            "//machina/core/protobuf:master_proto_cc",
            "//machina/core/protobuf:worker_proto_cc",
        ],
        otherwise = [
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc_headers_only",
            "//machina/core/protobuf:eager_service_proto_cc_headers_only",
            "//machina/core/protobuf:master_proto_cc_headers_only",
            "//machina/core/protobuf:worker_proto_cc_headers_only",
        ],
    ),
)

tf_py_strict_test(
    name = "python_api_info_test",
    srcs = ["python_api_info_test.py"],
    tags = ["no_pip"],
    deps = [
        ":_pywrap_python_api_info",
        ":constant_op",
        ":test_lib",
        "//machina/python/eager:context",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

cc_library(
    name = "python_api_dispatcher",
    srcs = ["python_api_dispatcher.cc"],
    hdrs = ["python_api_dispatcher.h"],
    deps = [
        "//machina/core:framework_lite",
        "//machina/core/platform:logging",
        "//machina/python/lib/core:py_util",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "//machina/python/util:cpp_python_util",
        "//machina/python/util:function_parameter_canonicalizer",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
    alwayslink = 1,
)

tf_python_pybind_extension(
    name = "_pywrap_python_api_dispatcher",
    srcs = ["python_api_dispatcher_wrapper.cc"],
    hdrs = [
        "python_api_dispatcher.h",
        "//machina/python/lib/core:safe_pyobject_ptr_required_hdrs",
    ],
    # This data is needed to add hermetic CUDA tools in python runfiles.
    data = if_cuda_tools(
        [
            "@cuda_nvcc//:ptxas",
            "@cuda_nvcc//:nvvm",
        ],
    ),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_python_api_dispatcher.pyi",
    ],
    deps = [
        "//machina/python/util:function_parameter_canonicalizer_hdrs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ] + if_pywrap([":python_api_dispatcher"]),
)

tf_py_strict_test(
    name = "python_api_dispatcher_test",
    srcs = ["python_api_dispatcher_test.py"],
    tags = [
        "no_pip",
        "no_windows",  # TODO(b/184424727): Enable this test on Windows.
    ],
    deps = [
        ":_pywrap_python_api_dispatcher",
        ":constant_op",
        ":tensor",
        ":test_lib",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:test",
    ],
)

cc_library(
    name = "python_tensor_converter",
    srcs = ["python_tensor_converter.cc"],
    hdrs = ["python_tensor_converter.h"],
    deps = [
        "//machina/c/eager:c_api",
        "//machina/core:protos_all_cc",
        "//machina/python/eager:pywrap_tfe_lib",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "//machina/python/util:cpp_python_util",
        "@com_google_absl//absl/strings",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

# Note: this target is only used by python_tensor_converter_test.
tf_python_pybind_extension(
    name = "_pywrap_python_tensor_converter",
    srcs = ["python_tensor_converter_wrapper.cc"],
    hdrs = [
        "python_tensor_converter.h",
        "//machina/c:headers",
        "//machina/c/eager:pywrap_required_hdrs",
        "//machina/c/experimental/ops:pywrap_required_hdrs",
        "//machina/core/common_runtime/eager:pywrap_required_hdrs",
        "//machina/core/distributed_runtime:pywrap_required_hdrs",
        "//machina/core/distributed_runtime/coordination:pywrap_required_hdrs",
        "//machina/core/distributed_runtime/eager:pywrap_required_hdrs",
        "//machina/python/eager:pywrap_required_hdrs",
        "//machina/python/lib/core:safe_pyobject_ptr_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime:pywrap_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime/coordination:pywrap_required_hdrs",
        "@local_xla//xla/tsl/python/lib/core:numpy_hdr",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_python_tensor_converter.pyi",
    ],
    starlark_only = True,
    deps = [
        "//machina/c:pywrap_required_hdrs",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_headers_lib",
        "//machina/core/config:flags_headers",
        "//third_party/py/numpy:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ] + if_static(
        extra_deps = [
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc",
            "//machina/core/protobuf:eager_service_proto_cc",
            "//machina/core/protobuf:master_proto_cc",
            "//machina/core/protobuf:worker_proto_cc",
        ],
        otherwise = [
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc_headers_only",
            "//machina/core/protobuf:eager_service_proto_cc_headers_only",
            "//machina/core/protobuf:master_proto_cc_headers_only",
            "//machina/core/protobuf:worker_proto_cc_headers_only",
        ],
    ),
)

tf_py_strict_test(
    name = "python_tensor_converter_test",
    srcs = ["python_tensor_converter_test.py"],
    tags = ["no_pip"],
    deps = [
        ":_pywrap_python_tensor_converter",
        ":constant_op",
        ":dtypes",
        ":indexed_slices",
        ":tensor",
        ":test_lib",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "ops",  # "ops" is already the name of a deprecated target
    srcs = ["ops.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    # "//third_party/mlperf:__subpackages__",
    # "//third_party/py/tf_agents:__subpackages__",
    # "//third_party/py/tf_slim:__subpackages__",
    # "//machina:internal",
    # "//machina_models:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":c_api_util",
        ":composite_tensor",
        ":device",
        ":dtypes",
        ":errors",
        ":op_callbacks",
        ":registry",
        ":stack",
        ":tensor",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tensor_util",
        ":traceable_stack",
        ":versions",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_machina",
        "//machina/python:pywrap_tfe",
        "//machina/python:tf2",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/eager:context",
        "//machina/python/eager:core",
        "//machina/python/eager:monitoring",
        "//machina/python/eager:record",
        "//machina/python/ops:control_flow_util",
        "//machina/python/ops:handle_data_util",
        "//machina/python/platform:tf_logging",
        "//machina/python/profiler:trace",
        "//machina/python/types:core",
        "//machina/python/types:internal",
        "//machina/python/util:compat",
        "//machina/python/util:decorator_utils",
        "//machina/python/util:deprecation",
        "//machina/python/util:function_utils",
        "//machina/python/util:lock_util",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_contextlib",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_stack",
        "//machina/python/util:traceback_utils",
        "//third_party/py/numpy",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "op_callbacks",
    srcs = ["op_callbacks.py"],
    visibility = visibility + ["//third_party/cloud_tpu/convergence_tools:__subpackages__"],
    deps = [
        "//machina/python/eager:context",
        "//machina/python/eager:execute",
    ],
)

cuda_py_strict_test(
    name = "op_callbacks_test",
    srcs = ["op_callbacks_test.py"],
    deps = [
        ":constant_op",
        ":dtypes",
        ":op_callbacks",
        ":ops",
        ":sparse_tensor",
        ":test_lib",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:test",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:script_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:variables",
        "//machina/python/util:compat",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "indexed_slices",
    srcs = ["indexed_slices.py"],
    visibility = visibility + [
        "//machina:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":composite_tensor",
        ":composite_tensor_gradient",
        ":dtypes",
        ":ops",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tensor_spec",
        ":tensor_util",
        ":type_spec",
        "//machina/core:protos_all_py",
        "//machina/python:tf2",
        "//machina/python/eager:context",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/saved_model:nested_structure_coder",
        "//machina/python/types:internal",
        "//machina/python/util:compat",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "indexed_slices_test",
    srcs = ["indexed_slices_test.py"],
    deps = [
        ":composite_tensor_gradient",
        ":constant_op",
        ":indexed_slices",
        "//machina/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "tensor_conversion",
    srcs = ["tensor_conversion.py"],
    deps = [
        ":tensor_conversion_registry",
        "//machina/python/framework:tensor",
        "//machina/python/util:deprecation",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "tensor_conversion_registry",
    srcs = ["tensor_conversion_registry.py"],
    # TODO(b/266747022): remove extra visibility
    visibility = visibility + ["//learning/brain/experimental:__subpackages__"],
    deps = [
        ":dtypes",
        "//machina/python/types:core",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "func_graph",
    srcs = ["func_graph.py"],
    deps = [
        ":auto_control_deps",
        ":composite_tensor",
        ":constant_op",
        ":dtypes",
        ":errors",
        ":indexed_slices",
        ":ops",
        ":tensor",
        ":type_spec",
        "//machina/core/function/capture:capture_container",
        "//machina/core/function/trace_type",
        "//machina/python/eager:context",
        "//machina/python/eager:execute",
        "//machina/python/eager/polymorphic_function:composite_tensor_utils",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/saved_model:save_context",
        "//machina/python/types:core",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_contextlib",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
        "//machina/python/util:variable_utils",
    ],
)

py_strict_library(
    name = "auto_control_deps",
    srcs = ["auto_control_deps.py"],
    deps = [
        ":auto_control_deps_utils",
        ":dtypes",
        ":indexed_slices",
        ":op_def_registry",
        ":ops",
        ":registry",
        ":sparse_tensor",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:control_flow_util",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/util:nest",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_decorator_py",
    ],
)

py_strict_library(
    name = "auto_control_deps_utils",
    srcs = ["auto_control_deps_utils.py"],
    deps = [
        ":dtypes",
        "//machina/python/util:object_identity",
    ],
)

tf_py_strict_test(
    name = "auto_control_deps_test",
    size = "small",
    srcs = ["auto_control_deps_test.py"],
    deps = [
        ":auto_control_deps",
        ":constant_op",
        ":dtypes",
        ":ops",
        ":test_lib",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_switch_case",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:resource_variable_ops_gen",
        "//machina/python/ops:script_ops",
        "//machina/python/ops:sendrecv_ops_gen",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:adam",
        "//machina/python/training:momentum",
    ],
)

py_strict_library(
    name = "config",
    srcs = ["config.py"],
    visibility = visibility + ["//machina:internal"],
    deps = [
        ":errors",
        "//machina/python/eager:context",
        "//machina/python/util:_pywrap_determinism",
        "//machina/python/util:_pywrap_tensor_float_32_execution",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "config_test",
    size = "small",
    srcs = ["config_test.py"],
    tags = [
        "multi_gpu",
        "no_pip",
    ],  # test_ops are not available in pip.
    deps = [
        ":config",
        ":constant_op",
        ":dtypes",
        ":errors",
        ":ops",
        ":test_lib",
        ":test_ops",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)

cuda_py_strict_test(
    name = "config_vgpu_test",
    size = "small",
    srcs = ["config_vgpu_test.py"],
    shard_count = 5,
    tags = [
        "gpu",
        "multi_gpu",
        "no_pip",
    ],  # test_ops are not available in pip.
    deps = [
        ":config",
        ":constant_op",
        ":errors",
        ":ops",
        ":test_lib",
        "//machina/python/eager:context",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)

tpu_py_strict_test(
    name = "config_test_tpu",
    size = "small",
    srcs = ["config_test.py"],
    # copybara:uncomment_begin
    # args = ["--tpu_use_tfrt=false"],
    # copybara:uncomment_end
    disable_tfrt = True,  # TODO(b/291666911): Update this test to call initialize_tpu_system
    main = "config_test.py",
    tags = ["no_pip"],  # test_ops are not available in pip.
    deps = [
        ":config",
        ":constant_op",
        ":dtypes",
        ":errors",
        ":ops",
        ":test_lib",
        ":test_ops",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)

py_strict_library(
    name = "random_seed",
    srcs = ["random_seed.py"],
    visibility = visibility + [
        "//machina:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":config",
        ":ops",
        "//machina/python/eager:context",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "registry",
    srcs = ["registry.py"],
    deps = [
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
    ],
)

py_strict_library(
    name = "smart_cond",
    srcs = ["smart_cond.py"],
    visibility = visibility + ["//machina_model_optimization:__subpackages__"],
    deps = [
        ":tensor",
        ":tensor_util",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_case",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "smart_cond_test",
    size = "small",
    srcs = ["smart_cond_test.py"],
    deps = [
        ":constant_op",
        ":dtypes",
        ":ops",
        ":smart_cond",
        ":test_lib",
        "//machina/python/client:session",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
    ],
)

py_strict_library(
    name = "sparse_tensor",
    srcs = ["sparse_tensor.py"],
    visibility = visibility + [
        "//machina:internal",
        "//machina_models:__subpackages__",
        "//third_party/mlperf:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":composite_tensor",
        ":constant_op",
        ":dtypes",
        ":ops",
        ":override_binary_operator",
        ":tensor",
        ":tensor_shape",
        ":tensor_spec",
        ":tensor_util",
        ":type_spec",
        ":type_spec_registry",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_machina",
        "//machina/python:tf2",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:sparse_ops_gen",
        "//machina/python/saved_model:nested_structure_coder",
        "//machina/python/types:internal",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "weak_tensor",
    srcs = ["weak_tensor.py"],
    deps = [
        ":composite_tensor_gradient",
        ":dtypes",
        ":errors",
        ":extension_type",
        ":ops",
        ":tensor",
        ":tensor_conversion_registry",
        ":tensor_spec",
        "//machina/python/eager:context",
        "//machina/python/types:core",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "weak_tensor_test",
    srcs = ["weak_tensor_test.py"],
    main = "weak_tensor_test.py",
    deps = [
        ":constant_op",
        ":dtypes",
        ":errors",
        ":extension_type",
        ":ops",
        ":tensor",
        ":test_lib",
        ":weak_tensor",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/module",
        "//machina/python/platform:test",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:save",
        "//machina/python/types:core",
    ],
)

py_strict_library(
    name = "composite_tensor",
    srcs = ["composite_tensor.py"],
    visibility = visibility,
    deps = [
        "//machina/python:pywrap_machina",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "none_tensor",
    srcs = ["none_tensor.py"],
    visibility = visibility + [
        "//learning/deepmind:__subpackages__",
        "//third_party/py/tf_keras:__subpackages__",
    ],
    deps = [
        ":composite_tensor",
        ":type_spec",
        ":type_spec_registry",
    ],
)

py_strict_test(
    name = "composite_tensor_test",
    srcs = ["composite_tensor_test.py"],
    main = "composite_tensor_test.py",
    deps = [
        ":composite_tensor",
        ":constant_op",
        ":dtypes",
        ":indexed_slices",
        ":sparse_tensor",
        ":test_lib",
        ":type_spec",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/platform:test",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "composite_tensor_gradient",
    srcs = ["composite_tensor_gradient.py"],
    deps = [
        ":composite_tensor",
        "//machina/python/util:nest",
        "@pypi_typing_extensions//:pkg",
    ],
)

py_strict_library(
    name = "extension_type",
    srcs = ["extension_type.py"],
    visibility = visibility,
    deps = [
        ":composite_tensor",
        ":dtypes",
        ":extension_type_field",
        ":immutable_dict",
        ":tensor",
        ":tensor_shape",
        ":type_spec",
        ":type_spec_registry",
        "//machina/core:protos_all_py",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:composite_tensor_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/saved_model:nested_structure_coder",
        "//machina/python/util:nest",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
        "@pypi_typing_extensions//:pkg",
    ],
)

py_strict_library(
    name = "extension_type_field",
    srcs = ["extension_type_field.py"],
    deps = [
        ":composite_tensor",
        ":dtypes",
        ":immutable_dict",
        ":ops",
        ":tensor",
        ":tensor_shape",
        ":type_spec",
        "//machina/python/util:type_annotations",
    ],
)

py_strict_library(
    name = "immutable_dict",
    srcs = ["immutable_dict.py"],
    deps = [],
)

py_strict_library(
    name = "tensor_shape",
    srcs = ["tensor_shape.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = [
    # "//smartass/brain/configure/python:__pkg__",
    # "//third_party/py/reverb:__subpackages__",
    # "//third_party/py/tf_agents:__subpackages__",
    # "//third_party/py/tf_slim:__subpackages__",
    # "//machina:internal",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        "//machina/core:protos_all_py",
        "//machina/core/function/trace_type",
        "//machina/python:tf2",
        "//machina/python/eager:monitoring",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:nested_structure_coder",
        "//machina/python/types:trace",
        "//machina/python/util:tf_export",
        "//machina/tools/docs:doc_controls",
    ],
)

pytype_strict_library(
    name = "type_spec",
    srcs = ["type_spec.py"],
    visibility = visibility + ["//third_party/py/machina_gnn:__subpackages__"],
    deps = [
        ":composite_tensor",
        ":dtypes",
        ":tensor_conversion_registry",
        ":tensor_shape",
        "//machina/core:protos_all_py",
        "//machina/core/function/trace_type",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:nested_structure_coder",
        "//machina/python/types:core",
        "//machina/python/types:internal",
        "//machina/python/types:trace",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:nest",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
        "//machina/tools/docs:doc_controls",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "type_spec_registry",
    srcs = ["type_spec_registry.py"],
    visibility = visibility + ["//third_party/py/machina_gnn:__subpackages__"],
    deps = [
        "//machina/python/types:internal",
    ],
)

pytype_strict_library(
    name = "type_utils",
    srcs = ["type_utils.py"],
    visibility = [
        # Limit the exposure of fulltype (which is a graph/C++ concept) to Python.
        "//machina/python/data/ops:__pkg__",
        "//machina/python/ops/ragged:__pkg__",
        "//machina/python/ops/structured:__pkg__",
    ],
    deps = [
        ":type_spec",
        "//machina/core:protos_all_py",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/ops/structured:structured_tensor",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:nest",
    ],
)

pytype_strict_library(
    name = "flexible_dtypes",
    srcs = ["flexible_dtypes.py"],
    deps = [
        ":composite_tensor",
        ":dtypes",
        ":indexed_slices",
        ":ops",
        ":tensor_shape",
        ":weak_tensor",
        "//machina/python/ops:variables",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "meta_graph",
    srcs = ["meta_graph.py"],
    visibility = visibility + ["//machina:internal"],
    deps = [
        ":byte_swap_tensor",
        ":error_interpolation",
        ":graph_io",
        ":importer",
        ":op_def_registry",
        ":ops",
        ":tensor",
        ":versions",
        "//machina/core:protos_all_py",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/eager:context",
        "//machina/python/lib/io:file_io",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
        "@pypi_packaging//:pkg",
    ],
)

pytype_strict_library(
    name = "framework_lib",
    srcs = ["framework_lib.py"],
    visibility = visibility + ["//machina:internal"],
    deps = [
        ":device",
        ":dtypes",
        ":importer",
        ":indexed_slices",
        ":load_library",
        ":ops",
        ":random_seed",
        ":sparse_tensor",
        ":tensor",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tensor_util",
    ],
)

pytype_strict_library(
    name = "graph_io",
    srcs = ["graph_io.py"],
    visibility = visibility + ["//machina:internal"],
    deps = [
        ":byte_swap_tensor",
        ":ops",
        "//machina/python/lib/io:file_io",
        "//machina/python/util:tf_export",
    ],
)

pytype_strict_library(
    name = "importer",
    srcs = ["importer.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + ["//machina:internal"],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":c_api_util",
        ":device",
        ":errors",
        ":function",
        ":op_def_registry",
        ":ops",
        ":tensor",
        "//machina/core:protos_all_py",
        "//machina/python:tf2",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/ops:control_flow_util",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ] + if_xla_available([
        "//machina/python:_pywrap_tfcompile",
    ]),
)

pytype_strict_library(
    name = "load_library",
    srcs = ["load_library.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + ["//machina:internal"],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":_pywrap_python_op_gen",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/eager:context",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ] + if_xla_available([
        "//machina/python:_pywrap_tfcompile",
    ]),
)

pytype_strict_library(
    name = "stack",
    srcs = ["stack.py"],
    visibility = visibility + ["//machina:internal"],
    deps = [
        "//machina/python/util:tf_contextlib",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "tensor",
    srcs = ["tensor.py"],
    visibility = visibility + [
        "//machina:internal",
        "//machina_models:__subpackages__",
        "//third_party/mlperf:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":common_shapes",
        ":dtypes",
        ":errors",
        ":op_callbacks",
        ":stack",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tensor_util",
        ":type_spec",
        ":type_spec_registry",
        "//machina/core:protos_all_py",
        "//machina/core/function/trace_type",
        "//machina/python:tf2",
        "//machina/python/eager:context",
        "//machina/python/eager:monitoring",
        "//machina/python/eager:record",
        "//machina/python/ops:handle_data_util",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:nested_structure_coder",
        "//machina/python/types:core",
        "//machina/python/types:internal",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "tensor_spec",
    srcs = ["tensor_spec.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//third_party/py/reverb:__subpackages__",
    # "//third_party/py/tf_agents:__subpackages__",
    # "//machina:internal",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":tensor",
    ],
)

py_strict_library(
    name = "tensor_util",
    srcs = ["tensor_util.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//third_party/py/tf_slim:__subpackages__",
    # "//machina:internal",
    # "//machina_models:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":dtypes",
        ":errors",
        ":tensor_shape",
        "//machina/core:protos_all_py",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/types:core",
        "//machina/python/types:internal",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//machina/python/util:numpy_compat",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "traceable_stack",
    srcs = ["traceable_stack.py"],
    visibility = ["//visibility:public"],
    deps = [
    ],
)

py_strict_library(
    name = "versions",
    srcs = ["versions.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "gpu_util",
    srcs = ["gpu_util.py"],
    deps = [],
)

pytype_strict_library(
    name = "test_lib",
    srcs = ["test_util.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = visibility + [
    # "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    # "//third_party/py/neural_structured_learning:__subpackages__",
    # "//third_party/py/tf_agents:__subpackages__",
    # "//third_party/py/tf_keras:__subpackages__",
    # "//machina:internal",
    # "//machina_model_optimization:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":_test_metrics_util",
        ":config",
        ":device",
        ":dtypes",
        ":errors",
        ":gpu_util",
        ":importer",
        ":indexed_slices",
        ":ops",
        ":random_seed",
        ":sparse_tensor",
        ":tensor",
        ":tensor_shape",
        ":tensor_util",
        ":tfrt_utils",
        ":versions",
        "//machina/core:protos_all_py",
        "//machina/core/config:flags_py",
        "//machina/python:pywrap_sanitizers",
        "//machina/python:tf2",
        "//machina/python/client:device_lib",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/client:session",
        "//machina/python/compat",
        "//machina/python/compat:v2_compat",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_util",
        "//machina/python/ops:control_flow_util_v2",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:script_ops",
        "//machina/python/ops:sparse_ops_gen",
        "//machina/python/ops:summary_ops_v2",
        "//machina/python/ops:sync_ops_gen",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/ops/ragged:ragged_tensor_value",
        "//machina/python/platform:_pywrap_stacktrace_handler",
        "//machina/python/platform:test",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:server_lib",
        "//machina/python/util:_pywrap_util_port",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:nest",
        "//machina/python/util:numpy_compat",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
        "//machina/python/util:traceback_utils",
        "//machina/python/util/protobuf",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@pypi_portpicker//:pkg",
    ],
)

# Including this as a dependency will result in tests using
# :test_lib to use XLA.
py_strict_library(
    name = "is_xla_test_true",
    srcs = ["is_xla_test_true.py"],
    visibility = visibility + [
        "//machina:internal",
        "//machina_models:__subpackages__",
        "//third_party/py/keras_cv:__subpackages__",
        "//third_party/py/tf_keras:__subpackages__",
    ],
)

# Including this as a dependency will result in tests using
# :test_lib to use MLIR.
py_strict_library(
    name = "is_mlir_bridge_test_true",
    srcs = ["is_mlir_bridge_test_true.py"],
    visibility = visibility + ["//machina:internal"],
)

# Including this as a dependency will result in tests using
# :test_lib to NOT use MLIR.
py_strict_library(
    name = "is_mlir_bridge_test_false",
    srcs = ["is_mlir_bridge_test_false.py"],
    visibility = visibility,
)

py_strict_library(
    name = "tfrt_utils",
    srcs = ["tfrt_utils.py"],
)

py_strict_library(
    name = "combinations",
    srcs = ["combinations.py"],
    visibility = visibility + ["//machina:internal"],
    deps = [
        ":ops",
        ":test_combinations_lib",
        "//machina/python:tf2",
        "//machina/python/eager:context",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "test_combinations_lib",
    srcs = ["test_combinations.py"],
    deps = [
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "test_combinations_test",
    srcs = ["test_combinations_test.py"],
    deps = [
        ":test_combinations_lib",
        "//machina/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "memory_checker",
    srcs = [
        "memory_checker.py",
        "python_memory_checker.py",
    ],
    visibility = visibility + ["//machina:internal"],
    deps = [
        ":_python_memory_checker_helper",
        "//machina/python/platform:tf_logging",
        "//machina/python/profiler:trace",
        "//machina/python/util:tf_inspect",
    ],
)

tf_python_pybind_extension(
    name = "_python_memory_checker_helper",
    srcs = ["python_memory_checker_helper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_python_memory_checker_helper.pyi",
    ],
    deps = [
        "@pybind11",
    ],
)

tf_py_strict_test(
    name = "constant_op_test",
    size = "small",
    srcs = ["constant_op_test.py"],
    main = "constant_op_test.py",
    deps = [
        ":constant_op",
        ":dtypes",
        ":importer",
        ":ops",
        "//machina/core:protos_all_py",
        "//machina/python/eager:def_function",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops/parallel_for:control_flow_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ] + tf_additional_xla_deps_py(),
)

tf_py_strict_test(
    name = "registry_test",
    size = "small",
    srcs = ["registry_test.py"],
    main = "registry_test.py",
    deps = [
        ":registry",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "errors_test",
    size = "small",
    srcs = ["errors_test.py"],
    main = "errors_test.py",
    tags = ["no_windows"],  # TODO(b/184424727): Enable this test on Windows.
    deps = [
        ":_errors_test_helper",
        ":c_api_util",
        ":errors",
        "//machina/core:protos_all_py",
        "//machina/python/lib/io:_pywrap_file_io",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "error_interpolation_test",
    size = "small",
    srcs = ["error_interpolation_test.py"],
    main = "error_interpolation_test.py",
    deps = [
        ":constant_op",
        ":dtypes",
        ":error_interpolation",
        ":errors",
        ":ops",
        ":test_lib",
        ":traceable_stack",
        "//machina/core:protos_all_py",
        "//machina/python/eager:def_function",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:script_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "subscribe_test",
    size = "small",
    srcs = ["subscribe_test.py"],
    main = "subscribe_test.py",
    deps = [
        ":constant_op",
        ":dtypes",
        ":ops",
        ":sparse_tensor",
        ":subscribe",
        ":test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:script_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "proto_test",
    size = "small",
    srcs = ["proto_test.py"],
    main = "proto_test.py",
    deps = [
        ":constant_op",
        ":ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "function_test",
    size = "medium",
    srcs = ["function_test.py"],
    shard_count = 10,
    tags = [
        "no_rocm",
        "optonly",
    ],
    deps = [
        ":constant_op",
        ":dtypes",
        ":errors",
        ":function",
        ":graph_to_function_def",
        ":ops",
        ":tensor_shape",
        ":test_lib",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:control_flow_ops_gen",
        "//machina/python/ops:functional_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:linalg_ops",
        "//machina/python/ops:logging_ops",
        "//machina/python/ops:logging_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:template",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "versions_test",
    size = "small",
    srcs = ["versions_test.py"],
    main = "versions_test.py",
    deps = [
        ":versions",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "importer_test",
    size = "small",
    srcs = ["importer_test.py"],
    main = "importer_test.py",
    tags = ["no_rocm"],
    deps = [
        ":constant_op",
        ":device",
        ":dtypes",
        ":function",
        ":importer",
        ":ops",
        ":tensor_shape",
        ":test_lib",
        ":test_ops",
        ":versions",
        "//machina/core:protos_all_py",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:cond",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "meta_graph_testdata",
    srcs = [
        "testdata/metrics_export_meta_graph.pb",
    ],
    visibility = if_oss(["//visibility:public"]),
)

tf_py_strict_test(
    name = "meta_graph_test",
    size = "small",
    srcs = ["meta_graph_test.py"],
    data = [":meta_graph_testdata"],
    main = "meta_graph_test.py",
    tags = [
        "no_pip",
        "no_windows",
    ],
    deps = [
        ":constant_op",
        ":dtypes",
        ":error_interpolation",
        ":function",
        ":meta_graph",
        ":ops",
        ":test_lib",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:metrics",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/training:queue_runner_impl",
    ],
)

cuda_py_strict_test(
    name = "node_file_writer_test",
    size = "small",
    srcs = ["node_file_writer_test.py"],
    # On Windows, fails in tempfile.TemporaryDirectory.cleanup() with error: "The directory is not
    # empty", so test is disabled on Windows.
    # TODO(reedwm): Debug this.
    tags = ["no_windows"],
    deps = [
        ":config",
        ":constant_op",
        ":dtypes",
        ":tensor_util",
        ":test_lib",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "traceable_stack_test",
    size = "small",
    srcs = ["traceable_stack_test.py"],
    main = "traceable_stack_test.py",
    deps = [
        ":test_lib",
        ":traceable_stack",
        "//machina/python/platform:test",
        "//machina/python/util:tf_inspect",
    ],
)

tf_gen_op_wrapper_py(
    name = "test_ops",
    out = "test_ops.py",
    api_def_srcs = [
        "//machina/core/api_def:base_api_def",
        "//machina/core/api_def:python_api_def",
    ],
    extra_py_deps = [
        "//machina/python:pywrap_tfe",
        "//machina/python/util:deprecation",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    deps = [":test_ops_kernels"],
)

tf_kernel_library(
    name = "test_ops_kernels",
    srcs = ["test_ops.cc"],
    hdrs = ["test_ops.h"],
    gpu_srcs = ["test_ops.cu.cc"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/time",
    ] + if_static(["//machina/core/platform:tensor_float_32_utils"]),
    alwayslink = 1,
)

tf_py_strict_test(
    name = "common_shapes_test",
    size = "small",
    srcs = ["common_shapes_test.py"],
    main = "common_shapes_test.py",
    deps = [
        ":common_shapes",
        ":tensor_shape",
        ":test_lib",
        "//machina/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "ops_test",
    size = "small",
    srcs = ["ops_test.py"],
    main = "ops_test.py",
    tags = ["no_pip"],
    deps = [
        ":composite_tensor",
        ":config",
        ":constant_op",
        ":device",
        ":dtypes",
        ":errors",
        ":function",
        ":indexed_slices",
        ":ops",
        ":sparse_tensor",
        ":tensor",
        ":tensor_conversion_registry",
        ":tensor_shape",
        ":tensor_util",
        ":test_lib",
        ":test_ops",
        ":type_spec",
        ":versions",
        "//machina/core:protos_all_py",
        "//machina/python/autograph/core:ag_ctx",
        "//machina/python/client:session",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:function",
        "//machina/python/eager:wrap_function",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_ops_gen",
        "//machina/python/ops:gradients",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:resources",
        "//machina/python/ops:special_math_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:test",
        "//machina/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

# Test that op_allowlist works with ops with namespaces
tf_gen_op_wrapper_py(
    name = "test_namespace_ops",
    out = "test_namespace_ops.py",
    extra_py_deps = [
        "//machina/python:pywrap_tfe",
        "//machina/python/util:deprecation",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
    op_allowlist = ["Namespace>TestStringOutput"],
    py_lib_rule = py_strict_library,
    deps = [
        ":test_ops_kernels",
    ],
)

tf_py_strict_test(
    name = "op_allowlist_namespace_test",
    size = "small",
    srcs = ["op_allowlist_namespace_test.py"],
    main = "op_allowlist_namespace_test.py",
    tags = ["no_pip"],  # Avoid a missing test dependency error
    deps = [
        ":test_namespace_ops",
        "//machina/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "python_op_gen_annotation_test",
    size = "small",
    srcs = ["python_op_gen_annotation_test.py"],
    main = "python_op_gen_annotation_test.py",
    deps = [
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "ops_enable_eager_test",
    size = "small",
    srcs = ["ops_enable_eager_test.py"],
    main = "ops_enable_eager_test.py",
    deps = [
        ":ops",
        "//machina/python/eager:context",
        "//machina/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "extension_type_test",
    size = "small",
    srcs = ["extension_type_test.py"],
    main = "extension_type_test.py",
    deps = [
        ":constant_op",
        ":dtypes",
        ":extension_type",
        ":extension_type_field",
        ":immutable_dict",
        ":ops",
        ":tensor",
        ":tensor_shape",
        ":test_lib",
        ":type_spec",
        ":type_spec_registry",
        ":type_utils",
        "//machina/core:protos_all_py",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute:mirrored_strategy",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:cond",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:save",
        "//machina/python/util:dispatch",
        "//machina/python/util:nest",
        "//machina/python/util:tf_inspect",
        "@absl_py//absl/testing:parameterized",
        "@pypi_typing_extensions//:pkg",
    ],
)

tf_py_strict_test(
    name = "extension_type_field_test",
    size = "small",
    srcs = ["extension_type_field_test.py"],
    main = "extension_type_field_test.py",
    tags = ["no_pip"],  # TODO(b/184565242) Enable pip once we add a module that depends on this.
    deps = [
        ":constant_op",
        ":dtypes",
        ":extension_type_field",
        ":tensor",
        ":tensor_shape",
        ":test_lib",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "immutable_dict_test",
    size = "small",
    srcs = ["immutable_dict_test.py"],
    main = "immutable_dict_test.py",
    deps = [
        ":immutable_dict",
        ":test_lib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "tensor_shape_test",
    size = "small",
    srcs = ["tensor_shape_test.py"],
    main = "tensor_shape_test.py",
    deps = [
        ":dtypes",
        ":tensor_shape",
        ":test_lib",
        "//machina/core:protos_all_py",
        "//machina/core/function/trace_type",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "type_spec_test",
    size = "small",
    srcs = ["type_spec_test.py"],
    main = "type_spec_test.py",
    deps = [
        ":dtypes",
        ":ops",
        ":tensor_shape",
        ":tensor_spec",
        ":test_lib",
        ":type_spec",
        ":type_spec_registry",
        ":type_utils",
        "//machina/core:protos_all_py",
        "//machina/core/function/trace_type",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:test",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "tensor_test",
    size = "small",
    srcs = ["tensor_test.py"],
    main = "tensor_test.py",
    deps = [
        ":constant_op",
        ":dtypes",
        ":ops",
        ":tensor",
        ":tensor_shape",
        ":test_lib",
        ":type_spec",
        ":type_utils",
        "//machina/core:protos_all_py",
        "//machina/core/function/trace_type",
        "//machina/python/eager:context",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "sparse_tensor_test",
    size = "small",
    srcs = ["sparse_tensor_test.py"],
    main = "sparse_tensor_test.py",
    deps = [
        ":dtypes",
        ":errors",
        ":ops",
        ":sparse_tensor",
        ":tensor",
        ":tensor_shape",
        ":test_lib",
        ":type_utils",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "device_spec_test",
    size = "small",
    srcs = ["device_spec_test.py"],
    main = "device_spec_test.py",
    deps = [
        ":device_spec",
        ":test_lib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "device_test",
    size = "small",
    srcs = ["device_test.py"],
    main = "device_test.py",
    deps = [
        ":device",
        ":device_spec",
        ":ops",
        ":test_lib",
        "//machina/python/eager:context",
        "//machina/python/ops:variables",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "random_seed_test",
    size = "small",
    srcs = ["random_seed_test.py"],
    main = "random_seed_test.py",
    deps = [
        ":random_seed",
        ":test_lib",
        "//machina/python/eager:context",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "tensor_util_test",
    size = "small",
    srcs = ["tensor_util_test.py"],
    main = "tensor_util_test.py",
    tags = [
        "no_windows",
    ],
    deps = [
        ":constant_op",
        ":dtypes",
        ":func_graph",
        ":indexed_slices",
        ":ops",
        ":sparse_tensor",
        ":tensor_shape",
        ":tensor_util",
        ":test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:shape_util",
        "//machina/python/ops:state_ops_gen",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "test_util_test",
    size = "small",
    srcs = ["test_util_test.py"],
    main = "test_util_test.py",
    tags = [
        "no_pip",
        "no_windows",
    ],
    deps = [
        ":combinations",
        ":config",
        ":constant_op",
        ":dtypes",
        ":errors",
        ":indexed_slices",
        ":ops",
        ":random_seed",
        ":tensor",
        ":test_lib",
        ":test_ops",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_sanitizers",
        "//machina/python/compat",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:test",
        "//machina/python/util/protobuf:compare_test_proto_py",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "memory_checker_test",
    size = "medium",
    srcs = ["memory_checker_test.py"],
    main = "memory_checker_test.py",
    shard_count = 8,
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
        "noasan",  # TODO(b/149948895): Re-enable.
        "nomsan",  # TODO(b/149948895): Re-enable.
        "not_run:arm",
        "notsan",  # TODO(b/149948895): Re-enable.
        "nozapfhahn",
    ],
    deps = [
        ":constant_op",
        ":ops",
        ":test_lib",
        # TODO(kkb): Find more appropriate place to add `memory_checker` as deps
        # Adding directly to `test_lib` caused a Colab binary size
        # regression b/149433910 .
        ":memory_checker",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "dtypes_test",
    size = "small",
    srcs = ["dtypes_test.py"],
    main = "dtypes_test.py",
    deps = [
        ":_dtypes",
        ":dtypes",
        ":test_lib",
        "//machina:machina_py",
        "//machina/core:protos_all_py",
        "//machina/core/function/trace_type",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "op_def_library_test",
    size = "small",
    srcs = ["op_def_library_test.py"],
    deps = [
        ":constant_op",
        ":dtypes",
        ":function",
        ":op_def_library",
        ":op_def_library_pybind",
        ":ops",
        ":tensor",
        ":tensor_shape",
        ":test_lib",
        "//machina/core:protos_all_py",
        "//machina/python/eager:def_function",
        "//machina/python/platform:test",
        "//machina/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "kernels_test",
    size = "small",
    srcs = ["kernels_test.py"],
    main = "kernels_test.py",
    deps = [
        ":kernels",
        ":test_lib",
        "//machina/python/platform:test",
    ],
)

tf_proto_library(
    name = "cpp_shape_inference_proto",
    srcs = ["cpp_shape_inference.proto"],
    protodeps = tf_additional_all_protos(),
    # TODO(b/74620627): remove when _USE_C_SHAPES is removed
    visibility = ["//machina:internal"],
)

tf_python_pybind_extension(
    name = "_errors_test_helper",
    srcs = ["errors_test_helper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_errors_test_helper.pyi",
    ],
    starlark_only = True,
    deps = [
        "//machina/c:tf_status_headers",
        "//machina/core/platform:status",
        "//machina/python/lib/core:pybind11_status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@pybind11",
    ],
)

tf_py_strict_test(
    name = "c_api_util_test",
    size = "small",
    srcs = ["c_api_util_test.py"],
    deps = [
        ":c_api_util",
        ":test_lib",
        "//machina/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "graph_util_test",
    size = "small",
    srcs = ["graph_util_test.py"],
    deps = [
        ":constant_op",
        ":dtypes",
        ":function",
        ":graph_util",
        ":ops",
        ":tensor_util",
        ":test_lib",
        "//machina/core:protos_all_py",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:state_ops_gen",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "convert_to_constants_test",
    size = "medium",
    srcs = ["convert_to_constants_test.py"],
    shard_count = 4,
    tags = ["no_rocm"],
    deps = [
        ":constant_op",
        ":convert_to_constants",
        ":dtypes",
        ":function",
        ":importer",
        ":ops",
        ":tensor_spec",
        ":test_lib",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:def_function",
        "//machina/python/grappler:tf_optimizer",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:cond_v2",
        "//machina/python/ops:control_flow_case",
        "//machina/python/ops:control_flow_switch_case",
        "//machina/python/ops:control_flow_v2_toggles",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:ref_variable",
        "//machina/python/ops:rnn",
        "//machina/python/ops:rnn_cell_impl",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/ops:while_v2",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:constants",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:loader",
        "//machina/python/saved_model:save",
        "//machina/python/saved_model:simple_save",
        "//machina/python/trackable:autotrackable",
        "//machina/python/training:saver",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
    ],
)

cuda_py_benchmark_test(
    name = "graph_building_benchmark",
    size = "medium",
    srcs = ["graph_building_benchmark.py"],
    main = "graph_building_benchmark.py",
    deps = [
        ":dtypes",
        "//machina/python/eager:context",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/platform:client_testlib",
    ],
)

tf_python_pybind_extension(
    name = "_proto_comparators",
    srcs = ["proto_comparators.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_proto_comparators.pyi",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:protobuf",
        "//machina/python/lib/core:pybind11_status",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ],
)

pyx_library(
    name = "fast_tensor_util",
    srcs = ["fast_tensor_util.pyx"],
    py_deps = ["//machina/python/util:compat"],
)

py_strict_test(
    name = "tf2_test",
    srcs = ["tf2_test.py"],
    deps = [
        ":combinations",
        "//machina/python:tf2",
        "//machina/python/compat:v2_compat",
        "//machina/python/data/kernel_tests:test_base",
        "//machina/python/platform:_pywrap_tf2",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "flexible_dtypes_test",
    srcs = ["flexible_dtypes_test.py"],
    tags = [
        "no_windows",  # TODO(b/286939592): Enable this test on Windows.
    ],
    deps = [
        ":constant_op",
        ":dtypes",
        ":extension_type",
        ":flexible_dtypes",
        ":ops",
        ":tensor",
        ":weak_tensor",
        "//machina/python/ops:variables",
        "//machina/python/ops:weak_tensor_test_util",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_python_pybind_extension(
    name = "_test_metrics_util",
    srcs = ["test_metrics_util.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_test_metrics_util.pyi",
    ],
    deps = [
        "//machina/core:framework",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "cpp_shape_inference_proto_py_pb2",
#     has_services = 0,
#     deps = [":cpp_shape_inference_proto"],
# )
# copybara:uncomment_end

py_strict_library(
    name = "summary_test_util",
    testonly = True,
    srcs = ["summary_test_util.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/lib/io:tf_record",
        "//machina/python/platform:gfile",
    ],
)

cc_library(
    name = "tensor_shape_casters",
    hdrs = ["tensor_shape_casters.h"],
    features = ["-parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "@pybind11",
    ],
)
