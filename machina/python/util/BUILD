# Tensorflow util package

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable", "tf_py_strict_test", "tf_python_pybind_extension")
load("//machina/core/platform:build_config.bzl", "tf_proto_library")  # @unused
load("//machina/core/platform:build_config_root.bzl", "if_pywrap", "if_static")

visibility = [
    "//engedu/ml/tf_from_scratch:__pkg__",
    "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    "//third_party/mlperf:__subpackages__",
    "//machina:internal",
    "//machina/compiler/tf2xla:__pkg__",
    "//machina/lite/toco/python:__pkg__",
    "//machina_models:__subpackages__",
    "//machina_model_optimization:__subpackages__",
    "//third_party/py/cleverhans:__subpackages__",
    "//third_party/py/launchpad:__subpackages__",
    "//third_party/py/reverb:__subpackages__",
    "//third_party/py/neural_structured_learning:__subpackages__",
    "//third_party/py/machina_examples:__subpackages__",
    "//third_party/py/tf_agents:__subpackages__",  # For benchmarks.
    "//third_party/py/tf_slim:__subpackages__",
    "//third_party/py/machina_docs:__subpackages__",
    "//third_party/py/tf_keras:__subpackages__",
]

util_subpackage_visibility = visibility + [
    # copybara:uncomment "//learning/brain:composite_tensor_allowlist",
    "//machina:__pkg__",
    "//third_party/py/machina_core:__subpackages__",
    "//third_party/py/tf_agents:__subpackages__",
    "//third_party/py/tfx:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

py_strict_library(
    name = "core",
    deps = [
        ":tf_contextlib",
        ":tf_decorator_py",
        ":tf_export",
        ":tf_inspect",
        ":tf_stack",
    ],
)

# TODO(mdan): Move this utility outside of TF.
cc_library(
    name = "kernel_registry",
    srcs = ["kernel_registry.cc"],
    hdrs = ["kernel_registry.h"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/log",
    ],
    alwayslink = 1,
)

tf_python_pybind_extension(
    name = "_pywrap_tfprof",
    srcs = ["tfprof_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tfprof.pyi",
    ],
    deps = if_pywrap(
        if_false = [
            "//machina/core/profiler/internal:print_model_analysis_hdr",
            "@pybind11",
        ],
        if_true = [
            "//machina/core:framework",
            "//machina/core/framework:reader_base",
            "//machina/core:lib_headers_for_pybind",
            "//machina/core/profiler/internal:print_model_analysis",
            "@local_xla//third_party/python_runtime:headers",
            "@com_google_absl//absl/strings",
            "@eigen_archive//:eigen3",
            "@pybind11",
        ],
    ),
)

tf_python_pybind_extension(
    name = "_pywrap_utils",
    srcs = ["util_wrapper.cc"],
    hdrs = ["util.h"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_utils.pyi",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_nest",
    srcs = ["nest_wrapper.cc"],
    hdrs = ["nest.h"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_nest.pyi",
    ],
    deps = [
        "//machina/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ] + if_pywrap([":cpp_nest"]),
)

cc_library(
    name = "cpp_nest",
    srcs = ["nest.cc"],
    hdrs = ["nest.h"],
    deps = [
        ":cpp_python_util",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/platform:stringpiece",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "@local_xla//third_party/python_runtime:headers",
    ],
    alwayslink = 1,
)

tf_python_pybind_extension(
    name = "_pywrap_kernel_registry",
    srcs = ["kernel_registry_wrapper.cc"],
    hdrs = ["kernel_registry.h"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_kernel_registry.pyi",
    ],
    deps = [
        "//machina/core:framework_headers_lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ] + if_pywrap([":kernel_registry"]),
)

tf_python_pybind_extension(
    name = "_pywrap_stat_summarizer",
    srcs = ["stat_summarizer_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_stat_summarizer.pyi",
    ],
    deps = [
        "//machina/core:protos_all_cc",
        "//machina/core/util:stat_summarizer",
        "@local_xla//xla/tsl/util:stats_calculator_portable",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tensor_float_32_execution",
    srcs = ["tensor_float_32.cc"],
    hdrs = ["//machina/core/platform:tensor_float_32_hdr"],
    compatible_with = get_compatible_with_portable(),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tensor_float_32_execution.pyi",
    ],
    deps = [
        "//machina/core/platform:tensor_float_32_hdr_lib",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_determinism",
    srcs = ["determinism.cc"],
    hdrs = ["//machina/core/util:determinism_hdr"],
    compatible_with = get_compatible_with_portable(),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_determinism.pyi",
    ],
    deps = [
        "//machina/core/util:determinism_hdr_lib",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_util_port",
    srcs = ["port_wrapper.cc"],
    hdrs = ["//machina/core/util:port_hdrs"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_util_port.pyi",
    ],
    deps = [
        "//machina/core/util:port",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_transform_graph",
    srcs = ["transform_graph_wrapper.cc"],
    hdrs = ["//machina/tools/graph_transforms:transform_graph_hdrs"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_transform_graph.pyi",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:lib_proto_parsing",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/python/lib/core:pybind11_status",
        "@com_google_absl//absl/status",
        "@pybind11",
    ] + if_pywrap(["//machina/tools/graph_transforms:transform_graph_lib"]),
)

tf_python_pybind_extension(
    name = "_pywrap_checkpoint_reader",
    srcs = ["py_checkpoint_reader_wrapper.cc"],
    hdrs = [
        "//machina/c:checkpoint_reader_hdrs",
        "//machina/c:headers",
        "//machina/c:safe_ptr_hdr",
        "//machina/c/eager:headers",
        "//machina/python/lib/core:ndarray_tensor_hdr",
        "//machina/python/lib/core:py_exception_registry_hdr",
        "//machina/python/lib/core:safe_pyobject_ptr_required_hdrs",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_checkpoint_reader.pyi",
    ],
    deps = [
        "//machina/compiler/tf2tensorrt:common_utils",
        "//machina/compiler/tf2tensorrt:trt_parameters",
        "//machina/core:framework",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:portable_gif_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/util/tensor_bundle",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_status",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "//third_party/py/numpy:headers",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ] + if_pywrap(["@com_google_absl//absl/strings"]),
)

cc_library(
    name = "cpp_python_util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "@com_google_absl//absl/memory",
        "@local_xla//third_party/python_runtime:headers",
    ],
)

tf_py_strict_test(
    name = "decorator_utils_test",
    srcs = ["decorator_utils_test.py"],
    deps = [
        ":decorator_utils",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "deprecation_test",
    srcs = ["deprecation_test.py"],
    deps = [
        ":deprecated_module",
        ":deprecation",
        ":tf_inspect",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:strict_mode",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "dispatch_test",
    srcs = ["dispatch_test.py"],
    deps = [
        ":deprecation",
        ":dispatch",
        ":nest",
        ":tf_export",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:extension_type",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_conversion",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:bitwise_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:proto_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/linalg:linear_operator_diag",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//machina/python/platform:tf_logging",
        "//machina/python/types:core",
    ],
)

tf_py_strict_test(
    name = "keyword_args_test",
    srcs = ["keyword_args_test.py"],
    deps = [
        ":keyword_args",
        "//machina/python/platform:client_testlib",
    ],
)

pytype_strict_library(
    name = "tf_export",
    srcs = ["tf_export.py"],
    compatible_with = get_compatible_with_portable(),
    # copybara:uncomment_begin(google-only)
    # visibility = [
    # "//third_party/py/machina_core:__subpackages__",
    # "//machina:__pkg__",
    # "//machina:__subpackages__",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":tf_decorator_py",
        ":tf_inspect",
    ],
)

pytype_strict_library(
    name = "tf_contextlib",
    srcs = ["tf_contextlib.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//learning/deepmind/research/language/translation/lm:__subpackages__",
        "//machina:__pkg__",
        "//machina:__subpackages__",
        "//third_party/py/machina_core:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [":tf_decorator_py"],
)

py_strict_library(
    name = "tf_decorator_py",
    srcs = ["tf_decorator.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//learning/deepmind/research/language/translation/lm:__subpackages__",
        "//machina:__pkg__",
        "//machina:__subpackages__",
        "//third_party/py/machina_core:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
)

tf_py_strict_test(
    name = "tf_export_test",
    srcs = ["tf_export_test.py"],
    deps = [
        ":tf_decorator_py",
        ":tf_export",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "vlog_test",
    srcs = ["vlog_test.py"],
    deps = [
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "tf_stack",
    srcs = ["tf_stack.py"],
    # TODO(mdan): Remove public visibility.
    visibility = ["//visibility:public"],
    deps = [
        ":_tf_stack",
        "//machina/core:protos_all_py",
    ],
)

tf_python_pybind_extension(
    name = "_tf_stack",
    srcs = ["tf_stack.cc"],
    hdrs = [
        "//machina/c:headers",
        "//machina/c/eager:headers",
        # Using header directly is required to avoid ODR violations.
        "stack_trace.h",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_tf_stack.pyi",
    ],
    deps = [
        "//machina/c:pywrap_required_hdrs",
        "//machina/compiler/tf2tensorrt:common_utils",
        "//machina/compiler/tf2tensorrt:trt_parameters",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:framework_lite",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_headers_lib",
        "//machina/core/framework:pywrap_required_hdrs",
        "//machina/core/platform:path",
        "//machina/core/platform:stack_frame",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/core/util:managed_stack_trace",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:mutex",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ] + if_static([
        ":stack_trace",
    ]),
)

tf_py_strict_test(
    name = "tf_stack_test",
    srcs = ["tf_stack_test.py"],
    deps = [
        ":tf_stack",
        "//machina/python/platform:client_testlib",
    ],
)

cc_library(
    name = "stack_trace",
    srcs = ["stack_trace.cc"],
    hdrs = ["stack_trace.h"],
    deps = [
        "//machina/core/platform:stack_frame",
        "//machina/core/util:managed_stack_trace",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

cc_library(
    name = "function_parameter_canonicalizer",
    srcs = ["function_parameter_canonicalizer.cc"],
    hdrs = ["function_parameter_canonicalizer.h"],
    deps = [
        "//machina/core/platform:logging",
        "//machina/core/platform:macros",
        "//machina/python/lib/core:py_util",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
    ],
)

cc_library(
    name = "function_parameter_canonicalizer_hdrs",
    textual_hdrs = ["function_parameter_canonicalizer.h"],
    deps = [
        "//machina/python/lib/core:safe_pyobject_ptr",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/types:span",
        "@local_xla//third_party/python_runtime:headers",
    ],
)

tf_python_pybind_extension(
    name = "_function_parameter_canonicalizer_binding_for_test",
    # testonly = True,
    srcs = ["function_parameter_canonicalizer_binding_for_test.cc"],
    hdrs = [
        "function_parameter_canonicalizer.h",
        "//machina/python/lib/core:safe_pyobject_ptr_required_hdrs",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_function_parameter_canonicalizer_binding_for_test.pyi",
    ],
    starlark_only = True,
    deps = [
        "//machina/core:lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/types:span",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ] + if_pywrap(
        if_true = [
            "//machina/compiler/tf2xla:tf2xla_opset",
            "//machina/python/lib/core:pybind11_lib",
        ],
    ),
)

tf_py_strict_test(
    name = "function_parameter_canonicalizer_test",
    srcs = ["function_parameter_canonicalizer_test.py"],
    tags = [
        "no_pip",  # b/168621686
        "no_windows",  # b/169275019
    ],
    deps = [
        ":_function_parameter_canonicalizer_binding_for_test",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "traceback_utils_test",
    size = "small",
    srcs = ["traceback_utils_test.py"],
    deps = [
        ":traceback_utils",
        "//machina/python/eager:def_function",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "object_identity_test",
    size = "small",
    srcs = ["object_identity_test.py"],
    deps = [
        ":nest",
        ":object_identity",
        "//machina/python/platform:client_testlib",
    ],
)

# Placeholder for intenal nest_test comments.
tf_py_strict_test(
    name = "nest_test",
    size = "small",
    srcs = ["nest_test.py"],
    main = "nest_test.py",
    deps = [
        ":nest_test_main_lib",
    ],
)

py_strict_library(
    name = "nest_test_main_lib",
    testonly = True,
    srcs = ["nest_test.py"],
    deps = [
        ":nest",
        ":nest_util",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "serialization_test",
    size = "small",
    srcs = ["serialization_test.py"],
    main = "serialization_test.py",
    deps = [
        ":serialization",
        "//machina/python/framework:tensor_shape",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "function_utils_test",
    srcs = ["function_utils_test.py"],
    deps = [
        ":function_utils",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "tf_contextlib_test",
    size = "small",
    srcs = ["tf_contextlib_test.py"],
    deps = [
        ":tf_contextlib",
        ":tf_decorator_py",
        ":tf_inspect",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "tf_decorator_test",
    size = "small",
    srcs = ["tf_decorator_test.py"],
    deps = [
        ":tf_decorator_py",
        ":tf_inspect",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "tf_should_use",
    srcs = ["tf_should_use.py"],
    deps = [
        ":tf_decorator_py",
        "//machina/python/eager:context",
        "//machina/python/framework:ops",
        "//machina/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "tf_should_use_test",
    size = "small",
    srcs = ["tf_should_use_test.py"],
    deps = [
        ":tf_should_use",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "tf_inspect_test",
    size = "small",
    srcs = ["tf_inspect_test.py"],
    deps = [
        ":tf_decorator_py",
        ":tf_inspect",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "example_parser_configuration",
    srcs = ["example_parser_configuration.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_util",
    ],
)

py_strict_library(
    name = "deprecation",
    srcs = ["deprecation.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = util_subpackage_visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":decorator_utils",
        ":is_in_graph_mode",
        ":tf_contextlib",
        ":tf_decorator_py",
        ":tf_inspect",
        "//machina/python/framework:strict_mode",
        "//machina/python/platform:tf_logging",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
        "//machina/tools/docs:doc_controls",
    ],
)

py_strict_library(
    name = "type_annotations",
    srcs = ["type_annotations.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "nest_util",
    srcs = ["nest_util.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":__init__",
        ":compat",
        ":custom_nest_protocol",
        "//machina/python/platform:tf_logging",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
        "@pypi_wrapt//:pkg",
        "//machina/python:pywrap_machina",
        ":_pywrap_utils",
    ],
)

py_strict_library(
    name = "tf_inspect",
    srcs = ["tf_inspect.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        ":tf_decorator_py",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "deprecated_module",
    srcs = ["deprecated_module.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":deprecated_module_new",
        ":deprecation",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "lazy_loader",
    srcs = ["lazy_loader.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = util_subpackage_visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        "//machina/python/platform:tf_logging",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
    ],
)

tf_py_strict_test(
    name = "lazy_loader_test",
    srcs = ["lazy_loader_test.py"],
    deps = [
        ":lazy_loader",
        ":tf_inspect",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "decorator_utils",
    srcs = ["decorator_utils.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "variable_utils",
    srcs = ["variable_utils.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":__init__",
        ":nest",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:ops",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
        ":_pywrap_utils",
    ],
)

py_strict_library(
    name = "compat",
    srcs = ["compat.py"],
    compatible_with = get_compatible_with_portable(),
    # copybara:uncomment_begin(google-only)
    # visibility = util_subpackage_visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
        ":tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "numpy_compat",
    srcs = ["numpy_compat.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py test because not all machina tests use machina.bzl's py test.
        "//machina/python:global_test_configuration",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "object_identity",
    srcs = ["object_identity.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        ":compat",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "keyword_args",
    srcs = ["keyword_args.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        ":decorator_utils",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "serialization",
    srcs = ["serialization.py"],
    visibility = util_subpackage_visibility + [
        "//machina_kfac/python/kernel_tests:__subpackages__",
    ],
    deps = [
        ":compat",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_shape",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
        "//third_party/py/numpy",
        "@pypi_wrapt//:pkg",
    ],
)

py_strict_library(
    name = "traceback_utils",
    srcs = ["traceback_utils.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        ":tf_decorator_py",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
        ":tf_export",
    ],
)

py_strict_library(
    name = "module_wrapper",
    srcs = ["module_wrapper.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":__init__",
        ":tf_decorator_py",
        ":tf_inspect",
        "//machina/python/eager:monitoring",
        "//machina/python/platform:tf_logging",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
        "//machina/tools/compatibility:all_renames_v2",
        ":fast_module_type",
    ],
)

py_strict_library(
    name = "function_utils",
    srcs = ["function_utils.py"],
    visibility = util_subpackage_visibility,
    deps = [
        ":tf_decorator_py",
        ":tf_inspect",
        "//machina/core:protos_all_py",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "lock_util",
    srcs = ["lock_util.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "is_in_graph_mode",
    srcs = ["is_in_graph_mode.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "dispatch",
    srcs = ["dispatch.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = util_subpackage_visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":tf_decorator_py",
        ":tf_inspect",
        ":traceback_utils",
        ":type_annotations",
        "//machina/python/framework:_pywrap_python_api_dispatcher",
        "//machina/python/framework:ops",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
        ":tf_export",
    ],
)

py_strict_library(
    name = "keras_deps",
    srcs = ["keras_deps.py"],
    compatible_with = get_compatible_with_portable(),
    # copybara:uncomment_begin(google-only)
    # visibility = util_subpackage_visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
        ":tf_export",
    ],
)

py_strict_library(
    name = "nest",
    srcs = ["nest.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = util_subpackage_visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":__init__",
        ":nest_util",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
        ":tf_export",
        ":_pywrap_utils",
        ":_pywrap_nest",
    ],
)

py_strict_library(
    name = "deprecated_module_new",
    srcs = ["deprecated_module_new.py"],
    compatible_with = get_compatible_with_portable(),
    visibility = util_subpackage_visibility,
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
    ],
)

py_strict_library(
    name = "all_util",
    srcs = ["all_util.py"],
    compatible_with = get_compatible_with_portable(),
    # copybara:uncomment_begin(google-only)
    # visibility = util_subpackage_visibility,
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":tf_inspect",
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
    ],
)

tf_py_strict_test(
    name = "lock_util_test",
    size = "small",
    srcs = ["lock_util_test.py"],
    main = "lock_util_test.py",
    deps = [
        ":lock_util",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_python_pybind_extension(
    name = "fast_module_type",
    srcs = ["fast_module_type.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "fast_module_type.pyi",
    ],
    deps = [
        "//machina/core/platform:logging",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_py_strict_test(
    name = "fast_module_type_test",
    srcs = ["fast_module_type_test.py"],
    deps = [
        ":fast_module_type",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "module_wrapper_test",
    size = "small",
    srcs = ["module_wrapper_test.py"],
    deps = [
        ":module_wrapper",
        ":tf_inspect",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
        "//machina/tools/compatibility:all_renames_v2",
    ],
)

tf_py_strict_test(
    name = "example_parser_configuration_test",
    size = "small",
    srcs = ["example_parser_configuration_test.py"],
    main = "example_parser_configuration_test.py",
    deps = [
        ":example_parser_configuration",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "type_annotations_test",
    size = "small",
    srcs = ["type_annotations_test.py"],
    deps = [
        ":type_annotations",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

filegroup(
    name = "util_hdr",
    srcs = ["util.h"],
)

tf_py_strict_test(
    name = "compat_test",
    srcs = ["compat_test.py"],
    deps = [
        ":compat",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "numpy_compat_test",
    srcs = ["numpy_compat_test.py"],
    deps = [
        ":numpy_compat",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "variable_utils_test",
    srcs = ["variable_utils_test.py"],
    deps = [
        ":nest",
        ":variable_utils",
        "//machina/python/eager:context",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "pywrap_xla_ops_test",
    srcs = ["pywrap_xla_ops_test.py"],
    tags = [
        "no_gpu",
        "no_pip",
    ],
    deps = [
        ":pywrap_xla_ops",
        "//machina/python/platform:test",
    ],
)

tf_python_pybind_extension(
    name = "pywrap_xla_ops",
    srcs = ["tf2xla_opset_wrapper.cc"],
    hdrs = if_pywrap(
        if_false = [
            "//machina/compiler/tf2xla:tf2xla_opset_hdrs",
        ],
        if_true = [],
    ),
    enable_stub_generation = True,
    pytype_srcs = [
        "pywrap_xla_ops.pyi",
    ],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ] + if_pywrap(["//machina/compiler/tf2xla:tf2xla_opset"]),
)

py_strict_library(
    name = "tf_decorator_export",
    srcs = ["tf_decorator_export.py"],
    deps = [
        ":tf_decorator_py",
        ":tf_export",
    ],
)

py_strict_library(
    name = "custom_nest_protocol",
    srcs = ["custom_nest_protocol.py"],
)
