#
# Public targets:
#  ":platform" - Low-level and platform-specific Python code.

load(
    "@local_xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_library")

# Placeholder: load py_proto_library
load(
    "//machina:machina.bzl",
    "VERSION",
    "cc_header_only_library",
    "if_google",
    "if_nccl",
    "if_oss",
    "if_windows",
    "if_xla_available",
    "tf_enable_mlir_bridge",
    "tf_python_pybind_static_deps",
    "tsl_async_value_deps",
)
load(
    "//machina:machina.default.bzl",
    "get_compatible_with_portable",
    "pybind_extension",
    "pywrap_aware_cc_import",
    "pywrap_aware_filegroup",
    "pywrap_aware_genrule",
    "pywrap_binaries",
    "pywrap_common_library",
    "pywrap_library",
    "pywrap_machina_macro",
    "tf_external_workspace_visible",
    "tf_monitoring_python_deps",
    "tf_pybind_cc_library_wrapper",
    "tf_python_pybind_extension",
)
load(
    "//machina/core/platform:build_config.bzl",
    "tf_additional_binary_deps",
    "tf_proto_library",
)
load(
    "//machina/core/platform:build_config_root.bzl",
    "if_pywrap",
    "if_static",
    "tf_additional_plugin_deps",
    "tf_additional_profiler_deps",
)

# TODO(mdan): Break into per-directory files.

visibility = [
    "//engedu/ml/tf_from_scratch:__pkg__",
    "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    "//third_party/mlperf:__subpackages__",
    "//machina:internal",
    "//machina/dtensor:dtensor-internal",
    "//machina/lite/toco/python:__pkg__",
    "//machina_models:__subpackages__",
    "//machina_model_optimization:__subpackages__",
    "//third_party/py/cleverhans:__subpackages__",
    "//third_party/courier:__subpackages__",
    "//third_party/py/courier:__subpackages__",
    "//third_party/py/lingvo:__subpackages__",
    "//third_party/py/reverb:__subpackages__",
    "//third_party/py/tensorfn:__subpackages__",
    "//third_party/py/neural_structured_learning:__subpackages__",
    "//third_party/py/machina_docs:__subpackages__",
    "//third_party/py/machina_examples:__subpackages__",
    "//third_party/py/machina_recommenders:__subpackages__",
    "//third_party/py/tf_agents:__subpackages__",  # For benchmarks.
    "//third_party/py/tf_slim:__subpackages__",
    "//third_party/py/tf_keras:__subpackages__",
    "//third_party/py/starcraft2:__subpackages__",
    "//third_party/py/machina_gnn:__subpackages__",
    "//third_party/py/machina_numerics:__subpackages__",
    "//third_party/py/machina_privacy:__subpackages__",
    "//third_party/reverb:__subpackages__",
    "//machina_minigo:__subpackages__",
    "//research/graph/fairness/inproc_fair_reg:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

# Description:

py_strict_library(
    name = "python",
    srcs = ["__init__.py"],
    visibility = [
        "//machina:__pkg__",
        "//machina/compiler/aot/tests:__pkg__",  # TODO(b/34059704): remove when fixed
        "//machina/lite/toco/python:__pkg__",  # TODO(b/34059704): remove when fixed
        "//machina/python/debug:__subpackages__",
        "//machina/python/tools:__pkg__",  # TODO(b/34059704): remove when fixed
        "//machina/tools/quantization:__pkg__",  # TODO(b/34059704): remove when fixed
    ],
    deps = [
        ":no_contrib",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:stateful_random_ops",
        "//machina/python/ops/structured:structured_ops",
        "//machina/python/tpu:tpu_noestimator",
    ],
)

# Some of the dependencies in keras_lib are possibly outdated or not generally
# useful, so a future task is to determine a good set of dependencies.
# keras_lib may be removed in the future, it should NOT be used as
# a direct dependency.
# TODO(b/199420795) investigate minimal set of dependencies ...
py_strict_library(
    name = "keras_lib",
    visibility = [
        "//machina:__pkg__",
        "//machina:internal",
        "//machina/python/keras:__subpackages__",
        "//machina/python/tools:__pkg__",
        "//machina/python/tools/api/generator:__pkg__",
        "//machina/tools/api/tests:__pkg__",
        "//machina/tools/compatibility/update:__pkg__",
        "//third_party/py/machina_privacy:__subpackages__",  # TODO(b/163395075): remove when fixed
    ],
    deps = [
        "//machina/python/feature_column:feature_column_py",
        "//machina/python/keras",
        "//machina/python/layers",
        "//machina/python/ops:rnn",
    ],
)

py_strict_library(
    name = "no_contrib",
    srcs = ["__init__.py"],
    visibility = [
        "//machina:__pkg__",
        "//machina/python/keras:__subpackages__",
        "//machina/python/tools:__pkg__",
        "//machina/python/tools/api/generator:__pkg__",
        "//machina/tools/api/tests:__pkg__",
        "//machina/tools/compatibility/update:__pkg__",
        "//third_party/py/machina_core:__subpackages__",
    ],
    deps = [
        ":_pywrap_py_exception_registry",
        ":_pywrap_quantize_training",
        ":distributed_framework_test_lib",
        ":keras_lib",
        ":proto_exports",
        ":pywrap_machina",
        ":pywrap_tfe",
        "//machina/compiler/mlir/quantization/machina/python:quantize_model",
        "//machina/compiler/mlir/machina_to_stablehlo/python:pywrap_machina_to_stablehlo",
        "//machina/core:protos_all_py",
        "//machina/dtensor/python:dtensor",
        "//machina/python/autograph",
        "//machina/python/autograph/utils:testing",
        "//machina/python/client",
        "//machina/python/client:_pywrap_events_writer",
        "//machina/python/client:device_lib",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/client:timeline",
        "//machina/python/compat",
        "//machina/python/compat:v2_compat",
        "//machina/python/compiler/mlir",
        "//machina/python/compiler/tensorrt:init_py",
        "//machina/python/compiler/xla:compiler_py",
        "//machina/python/data",
        "//machina/python/debug:debug_py",
        "//machina/python/distribute",
        "//machina/python/distribute:combinations",  # For tf.__internal__ API.
        "//machina/python/distribute:distribute_config",
        "//machina/python/distribute:strategy_combinations",  # For tf.__internal__,
        "//machina/python/distribute/experimental/rpc:rpc_ops",
        "//machina/python/dlpack",
        "//machina/python/eager:def_function",
        "//machina/python/eager:monitoring",
        "//machina/python/eager:remote",
        "//machina/python/framework:_pywrap_python_op_gen",
        "//machina/python/framework:_test_metrics_util",
        "//machina/python/framework:combinations",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:extension_type",
        "//machina/python/framework:flexible_dtypes",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:framework_lib",
        "//machina/python/framework:graph_io",
        "//machina/python/framework:graph_util",
        "//machina/python/framework:importer",
        "//machina/python/framework:kernels",
        "//machina/python/framework:load_library",
        "//machina/python/framework:meta_graph",
        "//machina/python/framework:subscribe",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_ops",  # TODO(b/183988750): Break testing code out into separate rule.
        "//machina/python/framework:weak_tensor",
        "//machina/python/grappler:tf_cluster",
        "//machina/python/grappler:tf_item",
        "//machina/python/grappler:tf_optimizer",
        "//machina/python/lib/io:file_io",
        "//machina/python/lib/io:python_io",
        "//machina/python/lib/io:tf_record",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:audio_ops_gen",
        "//machina/python/ops:bincount_ops",
        "//machina/python/ops:bitwise_ops",
        "//machina/python/ops:boosted_trees_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:clustering_ops",
        "//machina/python/ops:collective_ops",
        "//machina/python/ops:composite_tensor_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:cond_v2",
        "//machina/python/ops:confusion_matrix",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:control_flow_case",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:control_flow_switch_case",
        "//machina/python/ops:cudnn_rnn_ops_gen",
        "//machina/python/ops:filesystem_ops",
        "//machina/python/ops:functional_ops",
        "//machina/python/ops:gradient_checker",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:histogram_ops",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:initializers_ns",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:manip_ops",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:map_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:metrics",
        "//machina/python/ops:nccl_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:proto_ops",
        "//machina/python/ops:random_crop_ops",
        "//machina/python/ops:ref_variable",
        "//machina/python/ops:rnn_cell",
        "//machina/python/ops:rnn_ops_gen",
        "//machina/python/ops:script_ops",
        "//machina/python/ops:sendrecv_ops_gen",
        "//machina/python/ops:session_ops",
        "//machina/python/ops:sets",
        "//machina/python/ops:sets_impl",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:standard_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:stateful_random_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops:uniform_quant_ops_gen",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:weak_tensor_ops",
        "//machina/python/ops:weak_tensor_test_util",
        "//machina/python/ops:weights_broadcast_ops",
        "//machina/python/ops:while_loop",
        "//machina/python/ops:while_v2",
        "//machina/python/ops/distributions",
        "//machina/python/ops/distributions:bijector_test_util",
        "//machina/python/ops/distributions:identity_bijector",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg:linear_operator_test_util",
        "//machina/python/ops/linalg/sparse:sparse_py",
        "//machina/python/ops/losses",
        "//machina/python/ops/numpy_ops:np_config",
        "//machina/python/ops/numpy_ops:np_random",
        "//machina/python/ops/numpy_ops:numpy",
        "//machina/python/ops/parallel_for",
        "//machina/python/ops/ragged",
        "//machina/python/ops/signal",
        "//machina/python/ops/signal:dct_ops",
        "//machina/python/ops/signal:fft_ops",
        "//machina/python/ops/signal:mel_ops",
        "//machina/python/ops/signal:mfcc_ops",
        "//machina/python/ops/signal:reconstruction_ops",
        "//machina/python/ops/signal:shape_ops",
        "//machina/python/ops/signal:spectral_ops",
        "//machina/python/ops/signal:util_ops",
        "//machina/python/ops/signal:window_ops",
        "//machina/python/ops/structured:structured_ops",
        "//machina/python/platform:_pywrap_stacktrace_handler",
        "//machina/python/platform:app",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:flags",
        "//machina/python/platform:gfile",
        "//machina/python/platform:resource_loader",
        "//machina/python/platform:sysconfig",
        "//machina/python/platform:tf_logging",
        "//machina/python/profiler",
        "//machina/python/profiler:profiler_v2",
        "//machina/python/profiler:trace",
        "//machina/python/saved_model",
        "//machina/python/summary:summary_py",
        "//machina/python/summary/writer",
        "//machina/python/tools:module_util",
        "//machina/python/tools/api/generator:create_python_api",
        "//machina/python/tpu:_pywrap_sparse_core_layout",
        "//machina/python/tpu:datasets",
        "//machina/python/tpu:functional",
        "//machina/python/tpu:preempted_hook_py",
        "//machina/python/tpu:tpu_noestimator",
        "//machina/python/training:evaluation",
        "//machina/python/training:quantize_training",
        "//machina/python/training:saver",
        "//machina/python/training:saver_test_utils",
        "//machina/python/types:core",
        "//machina/python/types:data",
        "//machina/python/types:distribute",
        "//machina/python/types:trace",
        "//machina/python/user_ops:ops",
        "//machina/python/util:_pywrap_checkpoint_reader",
        "//machina/python/util:_pywrap_kernel_registry",
        "//machina/python/util:_pywrap_nest",
        "//machina/python/util:_pywrap_stat_summarizer",
        "//machina/python/util:_pywrap_tfprof",
        "//machina/python/util:_pywrap_transform_graph",
        "//machina/python/util:_pywrap_util_port",
        "//machina/python/util:_pywrap_utils",
        "//machina/python/util:all_util",
        "//machina/python/util:compat",
        "//machina/python/util:dispatch",
        "//machina/python/util:module_wrapper",
        "//machina/python/util:tf_decorator_export",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "core",
    visibility = [
        "//machina:__pkg__",
    ],
    deps = [
        "//machina/python/types:core",
        "//machina/python/util:core",
    ],
)

# This target should only be used for API generation.
py_strict_library(
    name = "modules_with_exports",
    srcs = ["modules_with_exports.py"],
    visibility = [
        "//machina:__pkg__",
        "//machina/python/tools/api/generator:__pkg__",
        "//machina/tools/compatibility/update:__pkg__",
        "//third_party/py/machina_core:__subpackages__",
    ],
    deps = [
        ":no_contrib",
        ":proto_exports",
        ":pywrap_machina",
        ":tf2",
        "//machina/core:protos_all_py",
        "//machina/core/function/trace_type",
        "//machina/python/checkpoint/sharding:sharding_policies",
        "//machina/python/checkpoint/sharding:sharding_util",
        "//machina/python/client",
        "//machina/python/client:device_lib",
        "//machina/python/client:timeline",
        "//machina/python/compat:v2_compat",
        "//machina/python/compiler/mlir",
        "//machina/python/compiler/xla",
        "//machina/python/compiler/xla:compiler_py",
        "//machina/python/data",
        "//machina/python/debug/lib:check_numerics_callback",
        "//machina/python/debug/lib:dumping_callback",
        "//machina/python/distribute",
        "//machina/python/distribute:combinations",
        "//machina/python/distribute:merge_call_interim",
        "//machina/python/distribute:multi_process_runner",
        "//machina/python/distribute:multi_worker_test_base",
        "//machina/python/distribute:parameter_server_strategy_v2",
        "//machina/python/distribute:sharded_variable",
        "//machina/python/distribute:strategy_combinations",
        "//machina/python/distribute/coordinator:cluster_coordinator",
        "//machina/python/distribute/experimental/rpc:rpc_ops",
        "//machina/python/distribute/failure_handling:failure_handling_lib",
        "//machina/python/distribute/failure_handling:preemption_watcher",
        "//machina/python/dlpack",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:monitoring",
        "//machina/python/eager:remote",
        "//machina/python/feature_column:feature_column_py",
        "//machina/python/framework:combinations",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:config",
        "//machina/python/framework:errors",
        "//machina/python/framework:extension_type",
        "//machina/python/framework:framework_lib",
        "//machina/python/framework:graph_util",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_combinations_lib",
        "//machina/python/framework:versions",
        "//machina/python/lib/io:file_io",
        "//machina/python/lib/io:python_io",
        "//machina/python/lib/io:tf_record",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:audio_ops_gen",
        "//machina/python/ops:bincount_ops",
        "//machina/python/ops:bitwise_ops",
        "//machina/python/ops:boosted_trees_ops_gen",
        "//machina/python/ops:clustering_ops_gen",
        "//machina/python/ops:composite_tensor_ops",
        "//machina/python/ops:cond_v2",
        "//machina/python/ops:cudnn_rnn_ops_gen",
        "//machina/python/ops:debug_ops_gen",
        "//machina/python/ops:filesystem_ops_gen",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:image_ops",
        "//machina/python/ops:initializers_ns",
        "//machina/python/ops:manip_ops",
        "//machina/python/ops:map_ops_gen",
        "//machina/python/ops:metrics",
        "//machina/python/ops:nn",
        "//machina/python/ops:random_crop_ops",
        "//machina/python/ops:rnn",
        "//machina/python/ops:rnn_cell",
        "//machina/python/ops:rnn_ops_gen",
        "//machina/python/ops:sendrecv_ops_gen",
        "//machina/python/ops:sets",
        "//machina/python/ops:standard_ops",
        "//machina/python/ops:stateful_random_ops",
        "//machina/python/ops:tpu_ops_gen",
        "//machina/python/ops:uniform_quant_ops_gen",
        "//machina/python/ops:while_v2",
        "//machina/python/ops/distributions",
        "//machina/python/ops/linalg",
        "//machina/python/ops/linalg/sparse:sparse_py",
        "//machina/python/ops/losses",
        "//machina/python/ops/numpy_ops:np_array_ops",
        "//machina/python/ops/numpy_ops:np_arrays",
        "//machina/python/ops/numpy_ops:np_config",
        "//machina/python/ops/numpy_ops:np_dtypes",
        "//machina/python/ops/numpy_ops:np_math_ops",
        "//machina/python/ops/numpy_ops:np_random",
        "//machina/python/ops/numpy_ops:np_utils",
        "//machina/python/ops/ragged",
        "//machina/python/ops/ragged:ragged_ops",
        "//machina/python/ops/signal",
        "//machina/python/ops/structured:structured_ops",
        "//machina/python/platform:app",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:flags",
        "//machina/python/platform:gfile",
        "//machina/python/platform:resource_loader",
        "//machina/python/platform:sysconfig",
        "//machina/python/platform:tf_logging",
        "//machina/python/profiler",
        "//machina/python/profiler:profiler_client",
        "//machina/python/profiler:profiler_v2",
        "//machina/python/profiler:trace",
        "//machina/python/saved_model",
        "//machina/python/summary:summary_py",
        "//machina/python/summary:tb_summary",
        "//machina/python/tpu:tpu_noestimator",
        "//machina/python/training",
        "//machina/python/training:quantize_training",
        "//machina/python/user_ops:ops",
        "//machina/python/util:all_util",
        "//machina/python/util:compat",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_contextlib",
        "//machina/python/util:tf_decorator_export",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tfcompile",
    srcs = ["tfcompile_wrapper.cc"],
    dynamic_deps = [":_pywrap_machina_internal.so"] + select({
        "//machina:macos": ["//machina:libmachina_framework.%s.dylib" % VERSION],
        "//conditions:default": ["//machina:libmachina_framework.so.%s" % VERSION],
        "//machina:windows": [],
    }),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tfcompile.pyi",
    ],
    static_deps = tf_python_pybind_static_deps(),
    deps = [
        "@pybind11",
        "//machina/compiler/aot:tfcompile_lib",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_status",
        # The headers here cannot be brought in via cc_header_only_library
        "//machina/compiler/aot:llvm_targets",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_quantize_training",
    srcs = [
        "//machina/python/training:quantize_training_wrapper.cc",
    ],
    hdrs = ["//machina/core/common_runtime:quantize_training_hdrs"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_quantize_training.pyi",
    ],
    deps = [
        "//machina/core:framework_headers_lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_headers_lib",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_proto",
        "//machina/python/lib/core:pybind11_status",
        "@com_google_absl//absl/strings",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

# TODO(yanhuasun): Move this back and the source file back to lib/core directory.
tf_python_pybind_extension(
    name = "_pywrap_py_exception_registry",
    srcs = ["py_exception_registry_wrapper.cc"],
    dynamic_deps = [":_pywrap_machina_internal.so"] + select({
        "//machina:macos": ["//machina:libmachina_framework.%s.dylib" % VERSION],
        "//conditions:default": ["//machina:libmachina_framework.so.%s" % VERSION],
        "//machina:windows": [],
    }),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_py_exception_registry.pyi",
    ],
    static_deps = tf_python_pybind_static_deps(),
    # Do not sort: core:py_exception_registry must come before platform:status
    deps = [
        "@com_google_absl//absl/container:fixed_array",
        "@pybind11",
        "@local_xla//third_party/python_runtime:headers",
        "//machina/c:tf_status_headers",
        "//machina/core:protos_all_cc",
        "//machina/python/lib/core:py_exception_registry",
        "//machina/core/platform:status",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_toco_api",
    srcs = [
        "lite/toco_python_api_wrapper.cc",
    ],
    hdrs = ["//machina/lite/toco/python:toco_python_api_hdrs"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_toco_api.pyi",
    ],
    deps = [
        "//machina/compiler/mlir/quantization/machina/python:py_function_lib",
        "//machina/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ] + if_pywrap(["//machina/lite/toco/python:toco_python_api"]),
)

# TODO(edloper): Remove unused dependency on safe_ptr.  (Blocker: there are
# targets that depend are relying on cpp_python_util to pull in safe_ptr's
# third_party/machina/c:c_api_no_xla dependency, which registers
# ops/gradients, rather than depending on it themselves.)
cc_header_only_library(
    name = "py_func_headers_lib",
    features = ["-parse_headers"],
    tags = ["no-ide"],
    deps = [
        "//machina/python/lib/core:py_func_lib",
    ],
)

cc_header_only_library(
    name = "python_op_gen_headers_lib",
    extra_deps = [
        "//machina/core:protos_all_cc",
    ],
    features = ["-parse_headers"],
    tags = ["no-ide"],
    deps = [
        "//machina/python/framework:python_op_gen",
    ],
)

py_strict_library(
    name = "extra_py_tests_deps",
    # extra_py_tests_deps is public to allow the use of tf_py_test or
    # cuda_py_test build rules for out-of-tree custom ops.
    # extra_py_tests_deps should NOT be used as dependency directly
    # in out-of-tree BUILD files.
    # Some of the dependencies in :keras_lib are possibly outdated
    # or not generally useful, so a future task is to determine a good
    # set of dependencies.
    # TODO(b/199420795) investigate minimal set of dependencies ...
    visibility = ["//visibility:public"],
    deps = [
        ":keras_lib",
        "//third_party/py/numpy",
        "@pypi_scipy//:pkg",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "distributed_framework_test_lib",
    deps = ["//machina/python/framework:test_lib"],
)

# Note: this is a heavyweight library specialized for TensorFlow graphs. Do not use for
# other purposes.

py_strict_library(
    name = "global_test_configuration",
    compatible_with = get_compatible_with_portable(),
    deps =
        tf_enable_mlir_bridge(),
)

# `tree.compat` requires visibility exception to test against `nest_test`
# to facilitate convergence between `tree.compat` and `nest`.

tf_proto_library(
    name = "protos_all",
    srcs = glob(
        ["**/*.proto"],
        exclude = [
            "//machina/python/util:compare_test_proto_src",
            "framework/cpp_shape_inference.proto",
        ],
    ),
    protodeps = ["//machina/python/training:checkpoint_state"],
    visibility = visibility,
)

py_strict_library(
    name = "pywrap_machina",
    srcs = [
        "pywrap_machina.py",
    ] + if_static(
        ["pywrap_dlopen_global_flags.py"],
        # Import will fail, indicating no global dlopen flags
        otherwise = [],
    ),  # b/153585257
    visibility = ["//visibility:public"],
    deps = [
        ":pywrap_machina_internal",
        "//machina/python/platform:self_check",
    ],
)

pywrap_machina_macro(
    name = "pywrap_machina_internal",
    srcs = ["pywrap_machina_internal.cc"],
    dynamic_deps = select({
        "//machina:macos": [
            "//machina:libmachina_cc.%s.dylib" % VERSION,
            "//machina:libmachina_framework.%s.dylib" % VERSION,
        ],
        "//conditions:default": [
            "//machina:libmachina_cc.so.%s" % VERSION,
            "//machina:libmachina_framework.so.%s" % VERSION,
        ],
        "//machina:windows": [],
    }),
    exports_filter = [
        "@pybind11",
        "@local_config_python//:python_headers",
        "@com_googlesource_code_re2//:__subpackages__",
    ] + if_windows([
        "//:__subpackages__",
        "@com_google_absl//:__subpackages__",
        "@com_google_protobuf//:__subpackages__",
        "@eigen_archive//:__subpackages__",
        "@local_tsl//tsl:__subpackages__",
        "@local_xla//xla:__subpackages__",
    ]),
    roots = [
        "//machina/python/lib/core:py_exception_registry",
    ],
    static_deps = [
        "@arm_neon_2_x86_sse//:__subpackages__",
        "@bazel_tools//:__subpackages__",
        "@boringssl//:__subpackages__",
        "@clog//:__subpackages__",
        "@com_github_cares_cares//:__subpackages__",
        "@com_github_googlecloudplatform_machina_gcp_tools//:__subpackages__",
        "@com_github_grpc_grpc//:__subpackages__",
        "@com_google_absl//:__subpackages__",
        "@com_google_googleapis//:__subpackages__",
        "@com_google_protobuf//:__subpackages__",
        "@com_googlesource_code_re2//:__subpackages__",
        "@compute_library//:__subpackages__",
        "@cpuinfo//:__subpackages__",
        "@curl//:__subpackages__",
        "@dlpack//:__subpackages__",
        "@eigen_archive//:__subpackages__",
        "@farmhash_archive//:__subpackages__",
        "@farmhash_gpu_archive//:__subpackages__",
        "@fft2d//:__subpackages__",
        "@flatbuffers//:__subpackages__",
        "@FP16//:__subpackages__",
        "@FXdiv//:__subpackages__",
        "@gemmlowp//:__subpackages__",
        "@gif//:__subpackages__",
        "@highwayhash//:__subpackages__",
        "@hwloc//:__subpackages__",
        "@icu//:__subpackages__",
        "@jsoncpp_git//:__subpackages__",
        "@libjpeg_turbo//:__subpackages__",
        "@llvm_openmp//:__subpackages__",
        "@toolchain-project//:__subpackages__",
        "@llvm_terminfo//:__subpackages__",
        "@llvm_zlib//:__subpackages__",
        "@local_config_cuda//:__subpackages__",
        "@local_config_git//:__subpackages__",
        "@local_config_nccl//:__subpackages__",
        "@local_config_python//:__subpackages__",
        "@local_config_rocm//:__subpackages__",
        "@local_config_tensorrt//:__subpackages__",
        "@local_execution_config_platform//:__subpackages__",
        "@mkl_dnn_acl_compatible//:__subpackages__",
        "@nccl_archive//:__subpackages__",
        "@nvtx_archive//:__subpackages__",
        "@onednn//:__subpackages__",
        "@org_sqlite//:__subpackages__",
        "@platforms//:__subpackages__",
        "@png//:__subpackages__",
        "@pthreadpool//:__subpackages__",
        "@pybind11//:__subpackages__",
        "@ruy//:__subpackages__",
        "@snappy//:__subpackages__",
        "@sobol_data//:__subpackages__",
        "@stablehlo//:__subpackages__",
        "@tf_runtime//:__subpackages__",
        "//:__subpackages__",
        "@upb//:__subpackages__",
        "@XNNPACK//:__subpackages__",
        "@zlib//:__subpackages__",
        "@local_tsl//tsl:__subpackages__",
        "@local_xla//xla:__subpackages__",
    ] + tsl_async_value_deps(),
    win_def_file = ":pywrap_machina_filtered_def_file",
    deps = [
        ":_pywrap_library_dependency_enforcer",
        "//machina/c:c_api",
        "//machina/c:c_api_experimental",
        "//machina/c:checkpoint_reader",
        "//machina/c:env",
        "//machina/c:kernels",
        "//machina/c:kernels_experimental",
        "//machina/c:logging",
        "//machina/c:ops",
        "//machina/c:python_api",
        "//machina/c:safe_ptr",
        "//machina/c:tf_status_helper",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/c/experimental/filesystem:filesystem_interface",
        "//machina/c/experimental/gradients",
        "//machina/c/experimental/gradients/tape",
        "//machina/c/experimental/ops",
        "//machina/c/experimental/stream_executor",
        "//machina/cc/saved_model:fingerprinting_impl",
        "//machina/cc/saved_model:loader_lite_impl",
        "//machina/cc/saved_model:metrics_impl",
        "//machina/compiler/mlir/lite/python:converter_python_api",
        "//machina/compiler/mlir/quantization/stablehlo/python:pywrap_quantization_lib_impl",
        "//machina/compiler/mlir/quantization/machina/python:quantize_model_cc_impl",
        "//machina/compiler/mlir/machina/c:mlir_c_api_registration",
        "//machina/compiler/mlir/machina_to_stablehlo/python:pywrap_machina_to_stablehlo_lib_impl",
        "//machina/compiler/tf2tensorrt:op_converter_registry_impl",
        "//machina/compiler/tf2xla:tf2xla_opset",
        "//machina/core:framework_internal_impl",
        "//machina/core:lib",
        "//machina/core:lib_internal_impl",
        "//machina/core:reader_base",
        "//machina/core/common_runtime:core_cpu_impl",
        "//machina/core/common_runtime/gpu:gpu_runtime_impl",
        "//machina/core/common_runtime/pluggable_device:pluggable_device_runtime_impl",
        "//machina/core/config:flag_defs",
        "//machina/core/config:flags",
        "//machina/core/data/service:dispatcher_client",
        "//machina/core/data/service:grpc_util",
        "//machina/core/data/service:py_utils",
        "//machina/core/data/service:server_lib",
        "//machina/core/debug",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/function/runtime_client:runtime_client_cc",
        "//machina/core/grappler:grappler_item",
        "//machina/core/grappler:grappler_item_builder",
        "//machina/core/grappler/clusters:cluster",
        "//machina/core/grappler/clusters:single_machine",
        "//machina/core/grappler/clusters:virtual_cluster",
        "//machina/core/grappler/costs:graph_memory",
        "//machina/core/grappler/graph_analyzer:graph_analyzer_tool",
        "//machina/core/grappler/optimizers:custom_graph_optimizer_registry_impl",
        "//machina/core/grappler/optimizers:meta_optimizer",
        "//machina/core/kernels:data_service_ops",
        "//machina/core/platform:stacktrace_handler",
        "//machina/core/profiler:profiler_impl",
        "//machina/core/profiler/internal:print_model_analysis",
        "//machina/core/tpu/kernels:sparse_core_layout",
        "//machina/core/util:determinism",
        "//machina/distribute/experimental/rpc/kernels:rpc_ops",
        "//machina/dtensor/cc:dtensor_device_cc",
        "//machina/dtensor/cc:tensor_layout",
        "//machina/lite/delegates/flex:delegate",
        "//machina/lite/kernels/shim:tf_kernel_shim",
        "//machina/lite/toco/python:toco_python_api",
        "//machina/python/client:tf_session_helper",
        "//machina/python/eager:pywrap_tfe_lib",
        "//machina/python/framework:op_def_util_cc",
        "//machina/python/framework:python_api_dispatcher",
        "//machina/python/framework:python_api_info",
        "//machina/python/framework:python_api_parameter_converter",
        "//machina/python/framework:python_op_gen",
        "//machina/python/framework:python_tensor_converter",
        "//machina/python/grappler:cost_analyzer_lib",
        "//machina/python/grappler:model_analyzer_lib",
        "//machina/python/lib/core:py_func_lib",
        "//machina/python/lib/core:pybind11_absl",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_proto",
        "//machina/python/lib/core:pybind11_status",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "//machina/python/util:cpp_nest",
        "//machina/python/util:cpp_python_util",
        "//machina/python/util:function_parameter_canonicalizer",
        "//machina/python/util:kernel_registry",
        "//machina/tools/graph_transforms:transform_graph_lib",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_xla//third_party/python_runtime:headers",
        "@local_xla//xla/backends/profiler/cpu:python_tracer",
        "@local_xla//xla/stream_executor:stream_executor_impl",
        "@local_xla//xla/tsl/profiler/rpc:profiler_server_impl",
        "@local_xla//xla/tsl/profiler/rpc/client:profiler_client_impl",
        "@local_xla//xla/tsl/python/lib/core:numpy",
    ] + select({
        "//machina/compiler/mlir/python:disable_mlir_config": [],
        "//conditions:default": [
            "//machina/compiler/mlir/python:mlir",
        ],
    }) + if_static([
        "//machina/core/platform:tensor_float_32_utils",
        "//machina/core/platform:enable_tf2_utils",
    ]) + if_google([
        "//base",
        "//machina/core/util/tensor_bundle",
    ]) + if_oss([
        "//machina/core/distributed_runtime/rpc:grpc_server_lib",
        "//machina/core/distributed_runtime/rpc:grpc_session",
    ]) + if_oss(if_cuda_is_configured([
        # TODO(tmorris): These dependencies are added to get the RPATHs for
        # nvidia standalone wheels into pywrap_machina_internal. We might be
        # able to remove this in the future, as these stubs should already
        # be brought in via other dependencies.
        "@local_xla//xla/tsl/cuda:cudnn",
        "@local_xla//xla/tsl/cuda:cufft",
        "@local_xla//xla/tsl/cuda:cupti",
        "@local_xla//xla/tsl/cuda:nccl_rpath",
    ])) + if_xla_available([
        "//machina/compiler/aot:tfcompile_lib",
    ]) + tf_monitoring_python_deps() + tf_additional_plugin_deps() + tf_additional_profiler_deps() + tf_additional_binary_deps(),
)

# ** Targets for Windows build (start) **
# We need the following targets to expose symbols from _pywrap_machina.dll

pywrap_aware_filegroup(
    name = "win_lib_files_for_exported_symbols",
    srcs = [
        "//machina/c:checkpoint_reader",  # checkpoint_reader
        "//machina/c:python_api",  # tf_session
        "//machina/c:safe_ptr",  # checkpoint_reader
        "//machina/c:tf_status_helper",  # tfe
        "//machina/cc/saved_model:fingerprinting_impl",  # SavedModel fingerprinting
        "//machina/cc/saved_model:metrics_impl",  # SavedModel metrics
        "//machina/compiler/jit:flags",  # tfe
        "//machina/compiler/jit:get_compiler_ir",  # tfe
        "//machina/compiler/mlir/lite/python:converter_python_api",  # converter
        "//machina/compiler/mlir/quantization/machina/python:quantize_model_cc_impl",  # quantization
        "//machina/compiler/mlir/machina_to_stablehlo/python:pywrap_machina_to_stablehlo_lib_impl",  # machina_to_stablehlo
        "//machina/compiler/tf2xla:tf2xla_opset",  # pywrap_xla_ops
        "//machina/core:framework_internal_impl",  # op_def_registry
        "//machina/core:lib_internal_impl",  # device_lib
        "//machina/core:op_gen_lib",  # tf_session
        "//machina/core/common_runtime:graph_constructor",  # tf_session
        "//machina/core/common_runtime:quantize_training",  # quantize_training
        "//machina/core/common_runtime:session_options",  # device_lib, tfe, tf_session
        "//machina/core/common_runtime:session_state",  # tf_session
        "//machina/core/common_runtime/eager:context",  # tfe
        "//machina/core/common_runtime/eager:eager_executor",  # tfe
        "//machina/core/common_runtime/eager:tensor_handle",  # tfe
        "//machina/core/config:flag_defs",  # flags_api
        "//machina/core/config:flags",  # flags_api
        "//machina/core/data/service:dispatcher_client",  # dispatcher_client
        "//machina/core/data/service:grpc_util",  # grpc_util
        "//machina/core/data/service:server_lib",  # server_lib
        "//machina/core/framework:attr_value_proto_cc",  # tf_text
        "//machina/core/framework:op_def_proto_cc",  # tf_text
        "//machina/core/function/runtime_client:runtime_client_cc",  # runtime_client_pybind
        "//machina/core/grappler:devices",  # tf_cluster
        "//machina/core/grappler:grappler_item",  # tf_item
        "//machina/core/grappler:grappler_item_builder",  # tf_item
        "//machina/core/grappler/clusters:cluster",  # tf_cluster
        "//machina/core/grappler/clusters:single_machine",  # tf_cluster
        "//machina/core/grappler/clusters:utils",  # tf_optimizer
        "//machina/core/grappler/clusters:virtual_cluster",  # tf_cluster
        "//machina/core/grappler/costs:analytical_cost_estimator",  # cost analyzer
        "//machina/core/grappler/costs:graph_memory",  # tf_cluster
        "//machina/core/grappler/costs:graph_properties",  # tf_item
        "//machina/core/grappler/costs:measuring_cost_estimator",  # tf_cluster
        "//machina/core/grappler/costs:op_level_cost_estimator",  # tf_cluster
        "//machina/core/grappler/costs:utils",  # tf_cluster
        "//machina/core/grappler/graph_analyzer:graph_analyzer_tool",  # graph_analyzer
        "//machina/core/grappler/optimizers:meta_optimizer",  # tf_optimizer
        "//machina/core/grappler/utils:topological_sort",  # tf_item
        "//machina/core/platform:cpu_feature_guard",  # cpu_feature_guard
        "//machina/core/platform:statusor",  # tfe
        "//machina/core/profiler/internal:print_model_analysis",  # tfprof
        "//machina/core/profiler/rpc/client:profiler_client_impl",  # profiler
        "//machina/core/tpu/kernels:sparse_core_layout",  # sparse_core_layouts
        "//machina/core/util:determinism",  # determinism
        "//machina/core/util:port",  # util_port
        "//machina/core/util/tensor_bundle",  # checkpoint_reader
        "//machina/dtensor/cc:dtensor_device_cc",  # DTensor
        "//machina/dtensor/cc:tensor_layout",  # DTensor
        "//machina/lite/toco/python:toco_python_api",  # toco
        "//machina/python/client:tf_session_helper",  # tf_session
        "//machina/python/eager:pywrap_tfe_lib",  # pywrap_tfe_lib
        "//machina/python/framework:op_def_util_cc",  # op_def_util
        "//machina/python/framework:python_api_dispatcher",  # python_api_dispatcher
        "//machina/python/framework:python_api_info",  # python_api_info
        "//machina/python/framework:python_api_parameter_converter",  # python_api_parameter_converter
        "//machina/python/framework:python_op_gen",  # python_op_gen
        "//machina/python/framework:python_tensor_converter",  # python_tensor_converter
        "//machina/python/grappler:cost_analyzer_lib",
        "//machina/python/grappler:model_analyzer_lib",  # model_analyzer
        "//machina/python/lib/core:ndarray_tensor",  # checkpoint_reader
        "//machina/python/lib/core:py_exception_registry",  # py_exception_registry
        "//machina/python/lib/core:py_func_lib",
        "//machina/python/lib/core:safe_pyobject_ptr",  # checkpoint_reader
        "//machina/python/util:cpp_nest",
        "//machina/python/util:cpp_python_util",
        "//machina/python/util:kernel_registry",
        "//machina/tools/graph_transforms:transform_graph_lib",  # transform_graph
        "@local_tsl//tsl/platform:tensor_float_32_utils",  # tensor_float_32
        "@local_tsl//tsl/profiler/lib:profiler_session_impl",  # profiler
        "@local_xla//xla/tsl/profiler/backends/cpu:traceme_recorder_impl",  # profiler
        "@local_xla//xla/tsl/profiler/rpc:profiler_server_impl",  # profiler
        "@local_xla//xla/tsl/profiler/rpc/client:profiler_client_impl",
        "@local_xla//xla/tsl/python/lib/core:ml_dtypes_lib",  # bfloat16, float8_e4m3fn, float8_e5m2
        "@local_xla//xla/tsl/python/lib/core:numpy",  # checkpoint_reader
    ] + select({
        "//machina/compiler/mlir/python:disable_mlir_config": [],
        "//conditions:default": [
            "//machina/compiler/mlir/python:mlir",  # mlir
        ],
    }) + if_xla_available([
        "//machina/compiler/aot:tfcompile_lib",  # tfcompile
        "@local_xla//xla:status_macros",  # tfcompile
        "@local_xla//xla/hlo/ir:hlo",  # tfcompile
        "@local_xla//xla/tsl/concurrency:async_value",  # tfcompile
    ]),
    visibility = ["//visibility:private"],
)

# Filter the DEF file to reduce the number of symbols to 64K or less.
# Note that we also write the name of the pyd file into DEF file so that
# the dynamic libraries of custom ops can find it at runtime.
pywrap_aware_genrule(
    name = "pywrap_machina_filtered_def_file",
    srcs = select({
        "//machina:windows": [
            ":pybind_symbol_target_libs_file",
            ":win_lib_files_for_exported_symbols",
            "//machina:machina_def_file",
            "//machina/tools/def_file_filter:symbols_pybind",
        ],
        "//conditions:default": [],
    }),
    outs = ["pywrap_machina_filtered_def_file.def"],
    cmd = select({
        "//machina:windows": """
              $(location @local_config_def_file_filter//:def_file_filter) \\
              --input $(location //machina:machina_def_file) \\
              --output $@ \\
              --target _pywrap_machina_internal.pyd \\
              --symbols $(location //machina/tools/def_file_filter:symbols_pybind) \\
              --lib_paths_file $(location :pybind_symbol_target_libs_file)
          """,
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    tools = ["@local_config_def_file_filter//:def_file_filter"],
    visibility = ["//visibility:public"],
)

# Write to a file a list of all cc_library targets that we need for exporting symbols on Windows.
pywrap_aware_genrule(
    name = "pybind_symbol_target_libs_file",
    srcs = [":win_lib_files_for_exported_symbols"],
    outs = ["pybind_symbol_target_libs_file.txt"],
    cmd = select({
        "//machina:windows": """
            for SRC in $(SRCS); do
              echo $$SRC | sed 's/third_party\\///g' >> $@
            done
          """,
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    visibility = ["//visibility:public"],
)

# Get the import library of _pywrap_machina_internal.pyd, platform-specific to Windows.
pywrap_aware_filegroup(
    name = "get_pywrap_machina_import_lib_file",
    srcs = [":_pywrap_machina_internal.so"],
    output_group = "interface_library",
)

pywrap_aware_cc_import(
    name = "_pywrap_machina_internal_linux",
    shared_library = "//machina/python:lib_pywrap_machina_internal.so",
    visibility = tf_external_workspace_visible(visibility),
)

pywrap_aware_cc_import(
    name = "_pywrap_machina_internal_macos",
    shared_library = "//machina/python:lib_pywrap_machina_internal.dylib",
    visibility = tf_external_workspace_visible(visibility),
)

pywrap_aware_cc_import(
    name = "_pywrap_machina_internal_windows",
    interface_library = "//machina/python:pywrap_machina_import_lib_file",
    shared_library = "//machina/python:_pywrap_machina_internal.dll",
    visibility = tf_external_workspace_visible(visibility),
)

# Rename the import library for _pywrap_machina_internal.pyd to _pywrap_machina_internal.lib
# (It was _pywrap_machina_internal.so.if.lib).
pywrap_aware_genrule(
    name = "pywrap_machina_import_lib_file",
    srcs = [":get_pywrap_machina_import_lib_file"],
    outs = ["_pywrap_machina_internal.lib"],
    cmd = select({
        "//machina:windows": "cp -f $< $@",
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    visibility = ["//visibility:public"],
)

# Create a cc_import rule for the import library of _pywrap_machina_internal.dll
# so that custom ops' dynamic libraries can link against it.
pywrap_aware_cc_import(
    name = "pywrap_machina_import_lib",
    interface_library = select({
        "//machina:windows": ":pywrap_machina_import_lib_file",
        "//conditions:default": "not_existing_on_unix.lib",  # Just a placeholder for Unix platforms
    }),
    system_provided = 1,
)

# ** Targets for Windows build (end) **

py_strict_library(
    name = "pywrap_mlir",
    srcs = ["pywrap_mlir.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_mlir",
        ":pywrap_machina",
        "//machina/python/eager:context",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_mlir",
    srcs = ["mlir_wrapper.cc"],
    hdrs = [
        "//machina/c:headers",
        "//machina/c:safe_ptr_hdr",
        "//machina/c/eager:headers",
        "//machina/compiler/mlir/python:pywrap_mlir_hdrs",
        "//machina/python/lib/core:safe_pyobject_ptr_required_hdrs",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_mlir.pyi",
    ],
    deps = [
        "//machina/compiler/tf2tensorrt:common_utils",
        "//machina/compiler/tf2tensorrt:trt_parameters",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_status",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "@com_google_absl//absl/strings",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ] + if_pywrap(["//machina/compiler/mlir/python:mlir"]),
)

py_strict_library(
    name = "pywrap_sanitizers",
    srcs = ["pywrap_sanitizers.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_sanitizers",
        ":pywrap_machina",
    ],
)

tf_python_pybind_extension(
    name = "flags_pybind",
    srcs = ["//machina/core/config:flags_api_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "flags_pybind.pyi",
    ],
    visibility = [
        "//machina/core/config:__subpackages__",
        "//machina/tools/pip_package:__subpackages__",
    ],
    deps = [
        "//machina/core:lib_headers_for_pybind",
        "//machina/core/config:flags_headers",
        "//machina/python/lib/core:pybind11_lib",
        "@com_google_absl//absl/types:optional",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_sanitizers",
    srcs = ["sanitizers_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_sanitizers.pyi",
    ],
    deps = [
        "@pybind11",
    ],
)

cc_library(
    name = "unified_api_pywrap_required_headers",
    textual_hdrs = [
        "//machina/python/lib/core:basic_hdrs",
        "@local_xla//xla/tsl/python/lib/core:basic_hdrs",
        "//machina/c:headers",
        "//machina/c:safe_ptr_hdr",
        "//machina/c/eager:headers",
        "//machina/c/eager:pywrap_required_hdrs",
        "//machina/c/experimental/ops:pywrap_required_hdrs",
        "//machina/c/experimental/gradients:pywrap_required_hdrs",
        "//machina/c/experimental/gradients/tape:pywrap_required_hdrs",
        "//machina/core/common_runtime/eager:pywrap_required_hdrs",
        "//machina/python/eager:pywrap_required_hdrs",
        "//machina/core/common_runtime:core_cpu_lib_headers",
        "//machina/core/public:session.h",
        "//machina/core/public:session_options.h",
    ],
    visibility = ["//machina/python/framework/experimental:__pkg__"],
    deps = [
        "//machina/c:pywrap_required_hdrs",
        "//machina/core/config:flags_headers",
        "//machina/core/framework:pywrap_required_hdrs",
        "//third_party/py/numpy:headers",
        "@local_xla//third_party/python_runtime:headers",
    ],
)

py_strict_library(
    name = "pywrap_tfe",
    srcs = ["pywrap_tfe.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tfe",
        ":pywrap_machina",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tfe",
    srcs = ["tfe_wrapper.cc"],
    hdrs = if_pywrap(
        if_false = [
            "//machina/c:headers",
            "//machina/c:safe_ptr_hdr",
            "//machina/c/eager:headers",
            "//machina/c/eager:pywrap_required_hdrs",
            "//machina/c/experimental/ops:pywrap_required_hdrs",
            "@local_xla//xla/tsl/distributed_runtime:pywrap_required_hdrs",
            "@local_xla//xla/tsl/distributed_runtime/coordination:pywrap_required_hdrs",
            "@local_xla//xla/tsl/python/lib/core:numpy_hdr",
            "//machina/core/common_runtime/eager:pywrap_required_hdrs",
            "//machina/core/distributed_runtime:pywrap_required_hdrs",
            "//machina/core/distributed_runtime/coordination:pywrap_required_hdrs",
            "//machina/core/distributed_runtime/eager:pywrap_required_hdrs",
            "//machina/python/eager:pywrap_required_hdrs",
            "//machina/python/lib/core:py_exception_registry_hdr",
            "//machina/python/lib/core:safe_pyobject_ptr_required_hdrs",
            "//machina/python/util:util_hdr",
        ],
    ),
    dynamic_deps = [":_pywrap_machina_internal.so"] + select({
        "//machina:macos": ["//machina:libmachina_framework.%s.dylib" % VERSION],
        "//conditions:default": ["//machina:libmachina_framework.so.%s" % VERSION],
        "//machina:windows": [],
    }),
    enable_stub_generation = True,
    linkopts = [],
    pytype_srcs = [
        "_pywrap_tfe.pyi",
    ],
    static_deps = tf_python_pybind_static_deps(),
    deps = [
        "//machina/c:pywrap_required_hdrs",
        "//machina/c/eager:tfe_tensorhandle_internal_hdrs_only",
        "//machina/compiler/jit:flags_headers_only",
        "//machina/compiler/jit:get_compiler_ir_hdrs_only",
        "//machina/core:framework_headers_lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_headers_lib",
        "//machina/core/config:flags_headers",
        "//machina/core/framework:pywrap_required_hdrs",
        "//machina/core/lib/llvm_rtti",
        "//machina/core/platform",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_status",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "//third_party/py/numpy:headers",
        "@local_xla//third_party/python_runtime:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@dlpack",
        "@pybind11",
        # copybara:uncomment "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ] + if_pywrap(
        if_true = ["//machina/python/eager:pywrap_tfe_lib"],
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

py_strict_library(
    name = "pywrap_tfe_monitoring_reader",
    testonly = True,
    srcs = ["pywrap_tfe_monitoring_reader.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tfe_monitoring_reader",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tfe_monitoring_reader",
    testonly = True,
    srcs = ["tfe_wrapper_monitoring_reader.cc"],
    hdrs = [
        "//machina/c/eager:headers_monitoring_reader",
        "//machina/c/eager:pywrap_headers_monitoring_reader",
        "//machina/python/util:util_hdr",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tfe_monitoring_reader.pyi",
    ],
    deps = [
        "//machina/c:c_api",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental_reader",
        "//machina/c/eager:tfe_tensorhandle_internal_hdrs_only",
        "//machina/compiler/jit:flags_headers_only",
        "//machina/compiler/jit:get_compiler_ir_hdrs_only",
        "//machina/core:framework_headers_lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_headers_lib",
        "//machina/core/config:flags_headers",
        "//machina/core/framework:pywrap_required_hdrs",
        "//machina/core/lib/llvm_rtti",
        "//machina/core/lib/monitoring:cell_reader",
        "//machina/core/platform",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_status",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "//third_party/py/numpy:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_parallel_device",
    srcs = [
        "//machina/c:headers",
        "//machina/c:safe_ptr_hdr",
        "//machina/c/eager:headers",
        "//machina/c/eager/parallel_device:headers",
        "//machina/c/eager/parallel_device:sources",
        "//machina/python/distribute/parallel_device:pywrap_parallel_device.cc",
        "//machina/python/lib/core:safe_pyobject_ptr_required_hdrs",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_parallel_device.pyi",
    ],
    visibility = [
        "//machina/python/distribute/parallel_device:__pkg__",
        "//machina/tools/pip_package:__subpackages__",
    ],
    deps = [
        "//machina/c:pywrap_required_hdrs",
        "//machina/c/eager:tfe_cancellationmanager_internal_hdrs_only",
        "//machina/c/eager:tfe_op_internal",
        "//machina/c/eager:tfe_tensorhandle_internal_hdrs_only",
        "//machina/core:framework_headers_lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:protos_all_cc",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_status",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_dtensor_device",
    srcs = ["pywrap_dtensor_device.cc"],
    data = [
        "_pywrap_dtensor_device.pyi",
    ],
    enable_stub_generation = True,
    features = ["-layering_check"],
    deps = [
        ":pywrap_densor_device_headers",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_status_headers",
        "@local_xla//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

tf_pybind_cc_library_wrapper(
    name = "pywrap_densor_device_headers",
    deps = [
        "//machina/c:c_api",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_internal",
        "//machina/core:protos_all_cc",
        "//machina/dtensor/cc:dtensor_device_cc",
        "//machina/dtensor/cc:tensor_layout",
        "//machina/dtensor/proto:layout_proto_cc",
        "//machina/python/eager:pywrap_tfe_lib",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "//machina/python/util:cpp_python_util",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
    ],
)

py_strict_library(
    name = "tf2",
    srcs = ["tf2.py"],
    deps = [
        "//machina/python/platform:_pywrap_tf2",
        "//machina/python/util:tf_export",
    ],
)

# copybara:uncomment_begin(google-only)
# # TODO(b/287346201): Look into replacing usages of this target.
# py_proto_library(
#     name = "protos_all_py_pb2",
#     has_services = 0,
#     deps = [":protos_all"],
# )
#
# # Special build rule for jax2tf.
# # TODO(b/239052279): remove once TF->TFLite dependency cycle is resolved
# py_strict_library(
#     name = "jax2tf_support",
#     visibility = ["//third_party/py/jax/experimental/jax2tf:__subpackages__"],
#     deps = [
#         "//machina/python/dlpack",
#         "//machina/python/ops/ragged:ragged_array_ops",
#         "//machina/python/saved_model",
#     ],
# )
# copybara:uncomment_end

pytype_strict_library(
    name = "proto_exports",
    srcs = ["proto_exports.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/util:tf_export",
    ],
)

pybind_extension(
    name = "_pywrap_machina_internal",
    srcs = ["pywrap_machina_internal.cc"],
    pywrap_only = True,
    deps = [
        ":_pywrap_library_dependency_enforcer",
        "@pybind11",
    ],
)

pybind_extension(
    name = "_pywrap_machina_cc_only",
    srcs = [],
    deps = [
        ":_protobuf_inline_symbols_enforcer",
        "//machina/compiler/mlir/machina/c:mlir_c_api_registration",
        "//machina/core/distributed_runtime:server_lib",
        "//machina/core/distributed_runtime/rpc:grpc_server_lib",
        "//machina/core/distributed_runtime/rpc:grpc_session",
        "//machina/core/kernels:data_service_ops",
        "//machina/core/kernels:reader_ops",
        "//machina/distribute/experimental/rpc/kernels:rpc_ops",
        "//machina/dtensor/cc:tensor_layout",
        "@local_xla//xla/backends/profiler/cpu:python_tracer",
    ],
)

cc_library(
    name = "_protobuf_inline_symbols_enforcer",
    srcs = ["protobuf_inline_symbols_enforcer.cc"],
    local_defines = select({
        "//machina:windows": [
            "PROTOBUF_USE_DLLS",
            "LIBPROTOBUF_EXPORTS",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//machina/compiler/mlir/quantization/machina:exported_model_proto_cc",
        "//machina/compiler/mlir/quantization/machina:quantization_options_proto_cc",
        "//machina/core/framework:attr_value_proto_cc",
        "//machina/core/framework:device_attributes_proto_cc",
        "//machina/core/framework:function_proto_cc",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/grappler/costs:op_performance_data_cc",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/dtensor/proto:layout_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc",
    ],
)

cc_library(
    name = "_pywrap_library_dependency_enforcer",
    srcs = ["pywrap_library_dependency_enforcer.cc"],
    hdrs = ["pywrap_library_dependency_enforcer.h"],
    alwayslink = True,
)

cc_library(
    name = "_cuda_deps_filter",
    srcs = [],
    deps = if_pywrap(
        if_true = (
            if_cuda_is_configured([
                "@local_xla//xla/tsl/cuda:cupti",
                "@local_xla//xla/tsl/cuda:nvshmem_stub",
            ]) + if_nccl([
                "@local_config_nccl//:nccl",
            ])
        ),
    ),
)

pywrap_library(
    name = "_pywrap_machina",
    # buildifier: disable=unsorted-dict-items
    # @unsorted-dict-items
    common_lib_def_files_or_filters = {
        "machina/python/_pywrap_machina_common": ":_pywrap_machina.def",
    },
    # buildifier: disable=unsorted-dict-items
    # @unsorted-dict-items
    common_lib_filters = {
        "machina/machina_framework": select({
            "//machina:windows": [],
            "//conditions:default": [
                "//machina:machina_framework_pywrap_filter",
                ":_cuda_deps_filter",
                "@com_google_absl//absl/debugging:leak_check",
                "@com_google_absl//absl/flags:flag",
                "@com_google_absl//absl/log:die_if_null",
                "@com_google_absl//absl/random",
            ],
        }),
        "machina/machina_cc": select({
            "//machina:windows": [],
            "//conditions:default": ["//machina:machina_cc_pywrap_filter"],
        }),
    },
    # buildifier: disable=unsorted-dict-items
    # @unsorted-dict-items
    common_lib_linkopts = {
        "machina/machina_framework": select({
            "//machina:windows": [],
            "//machina:macos": [
                "-lpthread",
                "-ldl",
                "-lm",
            ],
            "//conditions:default": [
                "-z defs",
                "-lpthread",
                "-ldl",
                "-lm",
            ],
        }),
        "machina/machina_cc": select({
            "//machina:windows": [
                "-DEFAULTLIB:ws2_32.lib",
                "-DEFAULTLIB:advapi32.lib",
                "-DEFAULTLIB:crypt32.lib",
                "-DEFAULTLIB:Normaliz.lib",
                "-DEFAULTLIB:ntdll.lib",
            ],
            "//machina:macos": [
                "-lpthread",
                "-ldl",
                "-lm",
            ],
            "//conditions:default": [
                "-z defs",
                "-lpthread",
                "-ldl",
                "-lm",
            ],
        }),
    },
    # buildifier: disable=unsorted-dict-items
    # @unsorted-dict-items
    common_lib_version_scripts = {
        "machina/machina_cc": select({
            "//machina:windows": None,
            "//machina:macos": "//machina:tf_exported_symbols.lds",
            "//conditions:default": "//machina:tf_version_script.lds",
        }),
    },
    # buildifier: disable=unsorted-dict-items
    # @unsorted-dict-items
    common_lib_versions = {
        "machina/machina_framework": "2",
        "machina/machina_cc": "2",
    },
    enable_common_lib_starlark_only_filter = select({
        "//machina:windows": False,
        "//machina:macos": False,
        "//conditions:default": True,
    }),
    starlark_only_deps = [
        "//machina/compiler/mlir/python/mlir_wrapper:filecheck_wrapper",
        "//machina/compiler/mlir/quantization/stablehlo/python:pywrap_quantization",
        "//machina/compiler/mlir/tfr:tfr_wrapper",
        "//machina/python/framework:_errors_test_helper",
        "//machina/python/framework/experimental:_math_ops",
        "//machina/python/framework/experimental:_nn_ops",
        "//machina/python/framework/experimental:_tape",
        "//machina/python/framework/experimental:_unified_api",
        "//machina/python/framework:_op_def_util",
        "//machina/python/framework:_py_context_manager",
        "//machina/python/framework:_pywrap_python_api_info",
        "//machina/python/framework:_pywrap_python_api_parameter_converter",
        "//machina/python/framework:_pywrap_python_tensor_converter",
        "//machina/python/grappler:_pywrap_cost_analyzer",
        "//machina/python/grappler:_pywrap_model_analyzer",
        "//machina/python/util:_function_parameter_canonicalizer_binding_for_test",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/mlir/lite/python:_pywrap_converter_api",
        "//machina/compiler/mlir/quantization/machina/python:pywrap_function_lib",
        "//machina/compiler/mlir/quantization/machina/python:pywrap_quantize_model",
        "//machina/compiler/mlir/machina_to_stablehlo/python:pywrap_machina_to_stablehlo",
        "//machina/compiler/tf2tensorrt:_pywrap_py_utils",
        "//machina/python:_pywrap_dtensor_device",
        "//machina/python:_pywrap_mlir",
        "//machina/python:_pywrap_parallel_device",
        "//machina/python:_pywrap_py_exception_registry",
        "//machina/python:_pywrap_quantize_training",
        "//machina/python:_pywrap_sanitizers",
        "//machina/python:_pywrap_machina_cc_only",
        "//machina/python:_pywrap_machina_internal",
        "//machina/python:_pywrap_tfcompile",
        "//machina/python:_pywrap_tfe",
        "//machina/python:_pywrap_toco_api",
        "//machina/python:flags_pybind",
        "//machina/python/autograph/impl/testing:pybind_for_testing",
        "//machina/python/client:_pywrap_debug_events_writer",
        "//machina/python/client:_pywrap_device_lib",
        "//machina/python/client:_pywrap_events_writer",
        "//machina/python/client:_pywrap_tf_session",
        "//machina/python/data/experimental/service:_pywrap_server_lib",
        "//machina/python/data/experimental/service:_pywrap_snapshot_utils",
        "//machina/python/data/experimental/service:_pywrap_utils_exp",
        "//machina/python/framework:_dtypes",
        "//machina/python/framework:_op_def_library_pybind",
        "//machina/python/framework:_op_def_registry",
        "//machina/python/framework:_proto_comparators",
        "//machina/python/framework:_python_memory_checker_helper",
        "//machina/python/framework:_pywrap_python_api_dispatcher",
        "//machina/python/framework:_pywrap_python_op_gen",
        "//machina/python/framework:_test_metrics_util",
        "//machina/python/grappler:_pywrap_tf_cluster",
        "//machina/python/grappler:_pywrap_tf_item",
        "//machina/python/grappler:_pywrap_tf_optimizer",
        "//machina/python/lib/core:_pywrap_py_func",
        "//machina/python/lib/io:_pywrap_file_io",
        "//machina/python/lib/io:_pywrap_record_io",
        "//machina/python/platform:_pywrap_cpu_feature_guard",
        "//machina/python/platform:_pywrap_stacktrace_handler",
        "//machina/python/platform:_pywrap_tf2",
        "//machina/python/profiler/internal:_pywrap_profiler",
        "//machina/python/profiler/internal:_pywrap_traceme",
        "//machina/python/saved_model:pywrap_saved_model",
        "//machina/python/tpu:_pywrap_sparse_core_layout",
        "//machina/python/tpu:_pywrap_tpu_embedding",
        "//machina/python/util:_pywrap_checkpoint_reader",
        "//machina/python/util:_pywrap_determinism",
        "//machina/python/util:_pywrap_kernel_registry",
        "//machina/python/util:_pywrap_nest",
        "//machina/python/util:_pywrap_stat_summarizer",
        "//machina/python/util:_pywrap_tensor_float_32_execution",
        "//machina/python/util:_pywrap_tfprof",
        "//machina/python/util:_pywrap_transform_graph",
        "//machina/python/util:_pywrap_util_port",
        "//machina/python/util:_pywrap_utils",
        "//machina/python/util:_tf_stack",
        "//machina/python/util:fast_module_type",
        "//machina/python/util:pywrap_xla_ops",
    ],
)

pywrap_common_library(
    name = "pywrap_machina_framework",
    dep = ":_pywrap_machina",
    filter_name = "machina_framework",
    visibility = ["//visibility:public"],
)

pywrap_common_library(
    name = "pywrap_machina_common",
    dep = ":_pywrap_machina",
    filter_name = "_pywrap_machina_common",
    visibility = ["//visibility:public"],
)

alias(
    name = "machina_common_framework",
    actual = select({
        "//conditions:default": ":pywrap_machina_framework",
        "//machina:macos": ":pywrap_machina_framework",
        "//machina:windows": ":pywrap_machina_common",
    }),
    visibility = ["//visibility:public"],
)

pywrap_binaries(
    name = "pywrap_machina_binaries",
    dep = ":_pywrap_machina",
)
