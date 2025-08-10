load("@local_xla//xla/tsl:tsl.bzl", "if_macos", "internal_visibility")
load("@local_xla//xla/tsl:tsl.default.bzl", "tsl_pybind_extension")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "get_compatible_with_portable", "tf_python_pybind_extension")
load("//machina/core/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/python/profiler:__subpackages__"],
    licenses = ["notice"],
)

py_strict_library(
    name = "flops_registry",
    srcs = ["flops_registry.py"],
    deps = [
        "//machina/python/framework:graph_util",
        "//machina/python/framework:ops",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "flops_registry_test",
    srcs = ["flops_registry_test.py"],
    deps = [
        ":flops_registry",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "model_analyzer_testlib",
    srcs = ["model_analyzer_testlib.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:rnn",
        "//machina/python/ops:rnn_cell",
        "//machina/python/ops:tensor_array_grad",
        "//machina/python/ops:variable_scope",
        "//machina/python/profiler:model_analyzer",
        "//machina/python/training:gradient_descent",
        "//machina/python/util:_pywrap_tfprof",
        "//machina/python/util:compat",
    ],
)

py_strict_test(
    name = "print_model_analysis_test",
    srcs = ["print_model_analysis_test.py"],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "run_metadata_test",
    srcs = ["run_metadata_test.py"],
    tags = [
        "no_gpu",  # b/138442728
        "no_pip",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":model_analyzer_testlib",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/profiler:model_analyzer",
        "//machina/python/profiler:option_builder",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_traceme",
    srcs = ["traceme_wrapper.cc"],
    copts = tf_profiler_copts(),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_traceme.pyi",
    ],
    visibility = [
        "//perftools/accelerators/xprof/xprofilez/integration_tests:__pkg__",
        "//machina/python:__pkg__",
        "//machina/python/profiler:__subpackages__",
        "//machina/tools/pip_package:__subpackages__",
    ],
    deps = [
        "@local_xla//xla/python/profiler/internal:traceme_wrapper",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_profiler",
    srcs = ["profiler_wrapper.cc"],
    copts = tf_profiler_copts(),
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_profiler.pyi",
    ],
    visibility = [
        "//machina/core/profiler:internal",
        "//machina/python:__pkg__",
        "//machina/python/eager:__pkg__",
        "//machina/python/profiler:__pkg__",
        "//machina/tools/pip_package:__subpackages__",
    ],
    deps = [
        ":profiler_pywrap_impl",
        "//machina/core/profiler/rpc:profiler_server_for_pybind",
        "//machina/python/lib/core:pybind11_status",
        "@com_google_absl//absl/status",
        "@org_xprof//xprof/convert:repository",
        "@org_xprof//xprof/convert:tool_options",
        "@org_xprof//xprof/convert:xplane_to_tools_data",
        "@pybind11",
    ],
)

cc_library(
    name = "python_hooks",
    hdrs = ["python_hooks.h"],
    compatible_with = get_compatible_with_portable(),
    copts = tf_profiler_copts() + ["-fexceptions"],
    features = ["-use_header_modules"],  # Incompatible with -fexceptions.
    visibility = ["//visibility:private"],
    deps = [
        "//machina/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/python/profiler/internal:python_hooks",
        "@pybind11",
    ],
    alwayslink = True,
)

cc_library(
    name = "profiler_pywrap_impl",
    srcs = ["profiler_pywrap_impl.cc"],
    hdrs = ["profiler_pywrap_impl.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/profiler/lib:profiler_session_for_pybind",
        "//machina/core/profiler/rpc:profiler_server_for_pybind",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
        "@local_tsl//tsl/profiler/lib:profiler_session",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/tsl/profiler/convert:xplane_to_trace_events",
        "@local_xla//xla/tsl/profiler/rpc/client:capture_profile",
        "@local_xla//xla/tsl/profiler/utils:session_manager",
    ],
)

tsl_pybind_extension(
    name = "_pywrap_profiler_plugin",
    srcs = ["pywrap_profiler_plugin.cc"],
    pytype_srcs = [
        "_pywrap_profiler_plugin.pyi",
    ],
    visibility = internal_visibility([
        "//machina/core/profiler:internal",
        "//machina/python/eager:__pkg__",
        "//machina/python/profiler:__pkg__",
    ]),
    deps = [
        "//machina/core:protos_all_cc_impl",
        "//machina/core/framework:attr_value_proto_cc_impl",
        "//machina/core/framework:op",
        "//machina/core/framework:tensor",
        "//machina/python/lib/core:py_exception_registry",
        "//machina/python/lib/core:pybind11_status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_protobuf//:protobuf",
        "@local_tsl//tsl/profiler/protobuf:profiler_analysis_proto_cc_impl",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc_impl",
        "@local_tsl//tsl/profiler/protobuf:profiler_service_monitor_result_proto_cc_impl",
        "@local_tsl//tsl/profiler/protobuf:profiler_service_proto_cc_impl",
        "@local_xla//xla:autotune_results_proto_cc_impl",
        "@local_xla//xla:autotuning_proto_cc_impl",
        "@local_xla//xla:xla_data_proto_cc_impl",
        "@local_xla//xla:xla_proto_cc_impl",
        "@local_xla//xla/pjrt:status_casters",
        "@local_xla//xla/service:hlo_proto_cc_impl",
        "@local_xla//xla/service:metrics_proto_cc_impl",
        "@local_xla//xla/service/gpu:backend_configs_cc_impl",
        "@local_xla//xla/service/gpu/model:hlo_op_profile_proto_cc_impl",
        "@local_xla//xla/stream_executor:device_description_proto_cc_impl",
        "@local_xla//xla/stream_executor/cuda:cuda_compute_capability_proto_cc_impl",
        "@local_xla//xla/tsl/framework:allocator_registry_impl",
        "@local_xla//xla/tsl/lib/io:table",
        "@local_xla//xla/tsl/platform:env_impl",
        "@local_xla//xla/tsl/platform:types",
        "@local_xla//xla/tsl/platform/cloud:gcs_file_system",
        "@local_xla//xla/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@local_xla//xla/tsl/profiler/rpc:profiler_server_impl",
        "@local_xla//xla/tsl/profiler/rpc:profiler_service_impl",
        "@local_xla//xla/tsl/profiler/rpc/client:capture_profile",
        "@local_xla//xla/tsl/profiler/rpc/client:profiler_client_impl",
        "@local_xla//xla/tsl/profiler/utils:session_manager",
        "@local_xla//xla/tsl/profiler/utils:time_utils_impl",
        "@local_xla//xla/tsl/protobuf:bfc_memory_map_proto_cc_impl",
        "@local_xla//xla/tsl/protobuf:dnn_proto_cc_impl",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc_impl",
        "@local_xla//xla/tsl/protobuf:histogram_proto_cc_impl",
        "@local_xla//xla/tsl/protobuf:rpc_options_proto_cc_impl",
        "@local_xla//xla/tsl/protobuf:test_log_proto_cc_impl",
        "@org_xprof//xprof/convert:tool_options",
        "@org_xprof//xprof/pywrap:profiler_plugin_impl",
        "@pybind11",
    ] + if_macos([
        "@local_xla//xla/tsl/lib/histogram",
        "@local_xla//xla/tsl/lib/io:record_writer",
        "@local_xla//xla/tsl/lib/monitoring:collection_registry",
        "@local_xla//xla/tsl/lib/monitoring:sampler",
        "//machina/core:framework_internal_impl",
        "//machina/core/common_runtime/gpu:gpu_id_impl",
        "//machina/core/lib/core:arena",
        "//machina/core/lib/strings:ordered_code",
        "//machina/core/platform:platform_strings",
        "@local_tsl//tsl/platform:random",
        "@local_xla//xla/tsl/platform:resource",
        "@local_tsl//tsl/profiler/lib:profiler_factory_impl",
        "@local_tsl//tsl/profiler/lib:profiler_session_impl",
    ]),
)
