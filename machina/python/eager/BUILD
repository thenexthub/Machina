load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("//machina:machina.bzl", "check_deps")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test", "tf_python_pybind_extension")
load("//machina/compiler/tests:build_defs.bzl", "tf_xla_py_strict_test")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_additional_rpc_deps",
)
load("//machina/python/tpu:tpu.bzl", "tpu_py_strict_test")
load(
    "//machina/tools/test:performance.bzl",
    "cuda_py_benchmark_test",
    "tf_py_logged_benchmark",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

cc_library(
    name = "pywrap_tfe_lib",
    srcs = [
        "pywrap_gradient_exclusions.cc",
        "pywrap_tensor.cc",
        "pywrap_tensor_conversion.cc",
        "pywrap_tfe_src.cc",
    ],
    hdrs = [
        "pywrap_gradient_exclusions.h",
        "pywrap_tensor.h",
        "pywrap_tensor_conversion.h",
        "pywrap_tfe.h",
    ],
    copts = ["-fexceptions"],
    features = [
        "-use_header_modules",  # Required for pybind11
    ],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/c:c_api",
        "//machina/c:c_api_internal",
        "//machina/c:safe_ptr",
        "//machina/c:tf_status_helper",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/c/eager:c_api_internal",
        "//machina/c/eager:dlpack",
        "//machina/c/eager:tape",
        "//machina/c/eager:tfe_context_internal",
        "//machina/c/eager:tfe_op_internal",
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime/eager:tensor_handle",
        "//machina/core/platform:logging",
        "//machina/core/platform:types",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/util:managed_stack_trace",
        "//machina/python/lib/core:ndarray_tensor",
        "//machina/python/lib/core:ndarray_tensor_bridge",
        "//machina/python/lib/core:py_exception_registry",
        "//machina/python/lib/core:py_seq_tensor",
        "//machina/python/lib/core:py_util",
        "//machina/python/lib/core:pybind11_status",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "//machina/python/util:cpp_python_util",
        "//machina/python/util:stack_trace",
        "//third_party/py/numpy:headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_xla//third_party/python_runtime:headers",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/python/lib/core:numpy",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "pywrap_tensor_test_util",
    testonly = True,
    srcs = ["pywrap_tensor_test_util.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "pywrap_tensor_test_util.pyi",
    ],
    deps = [
        ":pywrap_tfe_lib",
        "//machina/c:tf_status_helper",
        "//machina/c/eager:c_api_test_util",
        "//machina/python/lib/core:pybind11_lib",
        "@pybind11",
    ],
)

cuda_py_strict_test(
    name = "pywrap_tensor_test",
    size = "small",
    srcs = ["pywrap_tensor_test.py"],
    tags = [
        "no_oss",  # TODO(b/168051787): Enable.
        "no_pip",  # TODO(b/168051787): Enable.
    ],
    deps = [
        ":pywrap_tensor_test_util",
        ":test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "pywrap_tensor.h",
        "pywrap_tensor_conversion.h",
        "pywrap_tfe.h",
    ],
    visibility = ["//machina/python:__subpackages__"],
)

# Transitive dependencies of this target will be included in the pip package.
py_strict_library(
    name = "eager_pip",
    visibility = ["//machina:internal"],
    deps = [
        ":backprop",
        ":benchmarks_test_base",
        ":cancellation",
        ":context",
        ":core",
        ":def_function",
        ":execute",
        ":forwardprop",
        ":forwardprop_util",
        ":function",
        ":graph_only_ops",
        ":monitoring",
        ":tape",
        ":test",
        ":wrap_function",
        "//machina/python:pywrap_machina",
        "//machina/python/dlpack",
        "//machina/python/eager/memory_tests:memory_test_util",
    ],
)

py_strict_library(
    name = "core",
    srcs = ["core.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python:pywrap_tfe",
        "//machina/python/framework:errors",
        "//machina/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "cancellation",
    srcs = ["cancellation.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python:pywrap_tfe",
    ],
)

cuda_py_strict_test(
    name = "cancellation_test",
    size = "small",
    srcs = ["cancellation_test.py"],
    deps = [
        ":cancellation",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "executor",
    srcs = ["executor.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python:pywrap_tfe",
    ],
)

py_strict_library(
    name = "context",
    srcs = ["context.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = [
    # "//smartass/brain/ops:__subpackages__",
    # "//third_party/py/courier:__subpackages__",
    # "//third_party/py/jax_tpu_embedding:__subpackages__",
    # "//machina:internal",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":cancellation",
        ":execute",
        ":executor",
        ":monitoring",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_tfe",
        "//machina/python:tf2",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/framework:c_api_util",
        "//machina/python/framework:device",
        "//machina/python/framework:tfrt_utils",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:function_utils",
        "//machina/python/util:is_in_graph_mode",
        "//machina/python/util:tf_contextlib",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
        "@absl_py//absl/logging",
    ],
)

tf_python_pybind_extension(
    name = "custom_device_testutil",
    testonly = True,
    srcs = ["custom_device_testutil.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "custom_device_testutil.pyi",
    ],
    deps = [
        "//machina/c:c_api",
        "//machina/c:safe_ptr",
        "//machina/c:tf_status_helper",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/c/eager:custom_device_testutil",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_status",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "//machina/python/util:cpp_python_util",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

py_strict_test(
    name = "custom_device_test",
    size = "small",
    srcs = ["custom_device_test.py"],
    # Note that this currently only works with --config=monolithic, since it
    # requires the C API which runs static initializers again.
    #
    # TODO(allenl): Figure out a way to allow extensions to register custom
    # devices which works with dynamic linking.
    tags = [
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":context",
        ":custom_device_testutil",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "context_test",
    size = "small",
    srcs = ["context_test.py"],
    xla_enabled = True,
    deps = [
        ":context",
        ":def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
        "@local_xla//xla/service:hlo_proto_py",
    ],
)

tpu_py_strict_test(
    name = "context_tpu_platform_test",
    size = "small",
    srcs = ["context_tpu_platform_test.py"],
    tags = [
        "no_oss",
    ],
    deps = [
        ":def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "context_cross_platform_gpu_test",
    size = "small",
    srcs = ["context_cross_platform_gpu_test.py"],
    tags = [
        "no_oss",
        "noasan",  # TODO(b/417285186): Re-enable once the test is fixed.
    ],
    tpu_ops_enabled = True,
    xla_enabled = True,
    deps = [
        ":def_function",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "context_cross_platform_tpu_test",
    size = "small",
    srcs = ["context_cross_platform_tpu_test.py"],
    tags = [
        "no_oss",
    ],
    deps = [
        ":def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "monitoring",
    srcs = ["monitoring.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_tfe",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/framework:c_api_util",
        "//machina/python/util:compat",
        "//machina/python/util:tf_export",
    ],
)

cuda_py_strict_test(
    name = "monitoring_test",
    srcs = ["monitoring_test.py"],
    deps = [
        ":monitoring",
        ":test",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
    ],
)

cuda_py_strict_test(
    name = "small_constants_optimizer_test",
    size = "medium",
    srcs = ["small_constants_optimizer_test.py"],
    deps = [
        ":context",
        "//machina/python/eager/polymorphic_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "summary_optimizer_test",
    srcs = ["summary_optimizer_test.py"],
    tags = [
        "no_oss",
        "no_pip",
    ],  # TODO(b/219089812)
    deps = [
        "//machina/core:protos_all_py",
        "//machina/core/function/runtime_client:runtime_client_py",
        "//machina/python/data/ops:readers",
        "//machina/python/eager/polymorphic_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:summary_ops_v2",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "@absl_py//absl/flags",
    ],
)

py_strict_library(
    name = "tape",
    srcs = ["tape.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python:pywrap_tfe",
    ],
)

py_strict_library(
    name = "record",
    srcs = ["record.py"],
    visibility = ["//machina:internal"],
    deps = ["//machina/python:pywrap_tfe"],
)

cuda_py_strict_test(
    name = "tensor_test",
    srcs = ["tensor_test.py"],
    deps = [
        ":context",
        ":core",
        ":test",
        "//machina/python:pywrap_tfe",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:io_ops",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
    ],
)

cuda_py_strict_test(
    name = "backprop_test",
    srcs = ["backprop_test.py"],
    tags = [
        "no_cuda_asan",  # b/173825938
        "no_windows",  #TODO(b/139745667)
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":backprop",
        ":backprop_util",
        ":context",
        ":def_function",
        ":record",
        ":test",
        "//machina/python:pywrap_tfe",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:memory_checker",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:cond",
        "//machina/python/ops:custom_gradient",
        "//machina/python/ops:embedding_ops",
        "//machina/python/ops:functional_ops",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:gradients",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/training",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "forwardprop_test",
    srcs = ["forwardprop_test.py"],
    shard_count = 5,
    deps = [
        ":backprop",
        ":context",
        ":def_function",
        ":forwardprop",
        ":forwardprop_util",
        ":record",
        "//machina/python:pywrap_tfe",
        "//machina/python/distribute:mirrored_strategy",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:custom_gradient",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:map_fn",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_impl",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:unconnected_gradients",
        "//machina/python/ops:variables",
        "//machina/python/ops/parallel_for:control_flow_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "core_test",
    size = "small",
    srcs = ["core_test.py"],
    deps = [
        ":context",
        ":core",
        ":def_function",
        ":execute",
        ":executor",
        ":test",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_tfe",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:device",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:resource_variable_ops_gen",
        "//machina/python/ops:script_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "test",
    srcs = ["test.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python/framework:ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "execute",
    srcs = ["execute.py"],
    visibility = ["//machina:internal"],
    deps = [
        ":core",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_tfe",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_conversion_registry",
        "//machina/python/framework:tensor_shape",
        "//machina/python/types:core",
        "//machina/python/util:compat",
    ],
)

py_strict_library(
    name = "graph_only_ops",
    srcs = ["graph_only_ops.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:op_callbacks",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
    ],
)

cuda_py_strict_test(
    name = "graph_only_ops_test",
    srcs = ["graph_only_ops_test.py"],
    deps = [
        "graph_only_ops",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "framework_for_generated_wrappers",
    deprecation = "Depending on this target can cause build dependency cycles. Depend on the fine-grained sub-targets instead.",
    visibility = ["//visibility:public"],
    deps = [
        ":execute",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:tensor_shape",
    ],
)

py_strict_library(
    name = "function",
    srcs = ["function.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/eager/polymorphic_function:atomic_function",
        "//machina/python/eager/polymorphic_function:concrete_function",
        "//machina/python/eager/polymorphic_function:tf_method_target",
        "//machina/python/eager/polymorphic_function:tracing_compilation",
        "//machina/python/eager/polymorphic_function:transform",
    ],
)

py_strict_library(
    name = "backprop",
    srcs = ["backprop.py"],
    visibility = ["//machina:internal"],
    deps = [
        ":backprop_util",
        ":context",
        ":execute",
        ":imperative_grad",
        ":tape",
        "//machina/python:pywrap_tfe",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:composite_tensor_gradient",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:type_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:control_flow_util",
        "//machina/python/ops:default_gradient",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:unconnected_gradients",
        "//machina/python/ops/parallel_for:control_flow_ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:_pywrap_utils",
        "//machina/python/util:nest",
        "//machina/python/util:tf_contextlib",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
        "//machina/python/util:variable_utils",
    ],
)

py_strict_library(
    name = "backprop_util",
    srcs = ["backprop_util.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/core/config:flags_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:handle_data_util",
        "//machina/python/ops:math_ops",
    ],
)

py_strict_library(
    name = "forwardprop",
    srcs = ["forwardprop.py"],
    visibility = ["//machina:internal"],
    deps = [
        ":backprop",
        ":backprop_util",
        ":execute",
        ":forwardprop_util",
        "//machina/core/function/polymorphism:function_cache",
        "//machina/python:pywrap_tfe",
        "//machina/python/eager/polymorphic_function:tracing_compilation",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:unconnected_gradients",
        "//machina/python/ops/parallel_for:control_flow_ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "forwardprop_util",
    srcs = ["forwardprop_util.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python:pywrap_tfe",
    ],
)

py_strict_library(
    name = "benchmarks_test_base",
    srcs = ["benchmarks_test_base.py"],
    visibility = ["//machina:internal"],
    deps = [
        ":test",
        "//machina/python/platform:flags",
    ],
)

cuda_py_benchmark_test(
    name = "benchmarks_test",
    srcs = ["benchmarks_test.py"],
    deps = [
        ":backprop",
        ":benchmarks_test_base",
        ":context",
        ":core",
        ":def_function",
        ":forwardprop",
        ":test",
        "//machina/python:pywrap_tfe",
        "//machina/python/compat",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:cond",
        "//machina/python/ops:embedding_ops",
        "//machina/python/ops:functional_ops",
        "//machina/python/ops:gradients",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:math_ops_gen",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/util:nest",
        "//machina/python/util:tf_inspect",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "run_eager_op_as_function_test",
    srcs = ["run_eager_op_as_function_test.py"],
    tags = [
        "no_windows",  # b/207695287
    ],
    deps = [
        ":benchmarks_test_base",
        ":context",
        ":test",
        "//machina/python/data/experimental/ops:prefetching_ops",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:bitwise_ops",
        "//machina/python/ops:critical_section_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_map_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/util:tf_inspect",
    ],
)

tf_xla_py_strict_test(
    name = "run_eager_op_as_function_xla_test",
    size = "small",
    srcs = ["run_eager_op_as_function_xla_test.py"],
    enable_mlir_bridge = False,
    enabled_backends = [
        "cpu",
        "gpu",
    ],
    tags = [
        "multi_and_single_gpu",
        "no_pip",  # TODO(b/149738646): fix pip install so these tests run on kokoro pip
    ],
    deps = [
        ":def_function",
        ":test",
        "//machina/compiler/tests:xla_test",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
    ],
)

cuda_py_benchmark_test(
    name = "remote_benchmarks_test",
    srcs = ["remote_benchmarks_test.py"],
    deps = [
        ":context",
        ":def_function",
        ":remote",
        ":test",
        "//machina/python/framework:ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/training:server_lib",
    ],
)

tf_py_logged_benchmark(
    name = "benchmarks",
    target = "//machina/python/eager:benchmarks_test",
)

tf_py_strict_test(
    name = "record_test",
    srcs = ["record_test.py"],
    deps = [
        ":backprop",
        ":context",
        ":record",
        ":test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:custom_gradient",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:variables",
    ],
)

cuda_py_strict_test(
    name = "ops_test",
    srcs = ["ops_test.py"],
    deps = [
        ":context",
        ":execute",
        ":test",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:sparse_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "pywrap_tfe_test",
    srcs = ["pywrap_tfe_test.py"],
    deps = [
        ":backprop",
        ":context",
        ":core",
        ":def_function",
        ":test",
        "//machina/python:pywrap_tfe",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:test_ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:resource_variable_ops",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "imperative_grad",
    srcs = ["imperative_grad.py"],
    deps = [
        "//machina/python:pywrap_tfe",
        "//machina/python/ops:unconnected_gradients",
        "//machina/python/util:compat",
    ],
)

# Check that the main TensorFlow python library depends on and links XLA.
check_deps(
    name = "def_function_check_deps",
    required_deps = [
        "//machina/compiler/jit:xla_kernel_creator",
    ],
    deps = [":def_function"],
)

py_strict_library(
    name = "def_function",
    srcs = ["def_function.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = ["//machina:internal"],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        "//machina/python/eager/polymorphic_function",
        "//machina/python/eager/polymorphic_function:eager_function_run",
    ],
)

py_strict_library(
    name = "lift_to_graph",
    srcs = ["lift_to_graph.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python/framework:func_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:op_selector",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/util:compat",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "lift_to_graph_test",
    size = "medium",
    srcs = ["lift_to_graph_test.py"],
    deps = [
        "lift_to_graph",
        ":def_function",
        ":test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/util:compat",
    ],
)

py_strict_library(
    name = "wrap_function",
    srcs = ["wrap_function.py"],
    visibility = ["//machina:internal"],
    deps = [
        ":context",
        ":function",
        ":lift_to_graph",
        "//machina/core:protos_all_py",
        "//machina/core/function/polymorphism:function_type",
        "//machina/python/eager/polymorphic_function:atomic_function",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/platform:tf_logging",
        "//machina/python/saved_model:nested_structure_coder",
        "//machina/python/trackable:data_structures",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "wrap_function_test",
    srcs = ["wrap_function_test.py"],
    deps = [
        ":backprop",
        ":def_function",
        ":wrap_function",
        "//machina/core:protos_all_py",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:saver",
    ],
)

cuda_py_strict_test(
    name = "wrap_function_device_test",
    srcs = ["wrap_function_device_test.py"],
    xla_enable_strict_auto_jit = False,
    xla_enabled = False,
    deps = [
        ":def_function",
        ":wrap_function",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:importer",
        "//machina/python/framework:ops",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "remote",
    srcs = ["remote.py"],
    visibility = ["//machina:internal"],
    deps = [
        ":context",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_tfe",
        "//machina/python/distribute:device_util",
        "//machina/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//machina/python/framework:ops",
        "//machina/python/platform:remote_utils",
        "//machina/python/training:server_lib",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "@absl_py//absl/logging",
    ] + tf_additional_rpc_deps(),
)

cuda_py_strict_test(
    name = "remote_test",
    size = "medium",
    srcs = ["remote_test.py"],
    grpc_enabled = True,
    shard_count = 2,
    tags = [
        "no_oss",  # This test launches local server.
        "nofastbuild",  # times out
        "optonly",  # times out
    ],
    deps = [
        ":cancellation",
        ":context",
        ":def_function",
        ":executor",
        ":remote",
        ":test",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/distribute/cluster_resolver:base_cluster_resolver_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:test_ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:functional_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/training:server_lib",
        "//machina/python/util:compat",
        "@absl_py//absl/testing:parameterized",
        "@pypi_portpicker//:pkg",
    ],
)

cuda_py_strict_test(
    name = "remote_execution_test",
    srcs = ["remote_execution_test.py"],
    grpc_enabled = True,
    shard_count = 8,
    tags = [
        "no_oss",  # This test launches local server
    ],
    deps = [
        ":backprop",
        ":context",
        ":def_function",
        ":remote",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_tfe",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:script_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:server_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "remote_cluster_test",
    srcs = ["remote_cluster_test.py"],
    grpc_enabled = True,
    shard_count = 8,
    tags = [
        "no_oss",  # This test launches local server
        "no_tfrt",  # TODO(b/171765113)
        "notsan",  # TODO(b/170783249)
    ],
    deps = [
        ":context",
        ":def_function",
        ":executor",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_tfe",
        "//machina/python/distribute/failure_handling:check_preemption_py",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:coordinator",
        "//machina/python/training:server_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "remote_cloud_tpu_test",
    srcs = ["remote_cloud_tpu_test.py"],
    tags = [
        "notap",
    ],
    deps = [
        ":remote",
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/framework:config",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
    ],
)

tpu_py_strict_test(
    name = "remote_cloud_tpu_pod_test",
    srcs = ["remote_cloud_tpu_test.py"],
    args = ["--num_tpu_devices=32"],
    main = "remote_cloud_tpu_test.py",
    tags = [
        "notap",
        "tpu_pod",
    ],
    deps = [
        ":remote",
        "//machina/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//machina/python/framework:config",
        "//machina/python/tpu:tpu_strategy_util",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:absltest",
    ],
)

cuda_py_strict_test(
    name = "device_placement_test",
    size = "small",
    srcs = ["device_placement_test.py"],
    shard_count = 5,
    deps = [
        ":context",
        ":def_function",
        ":remote",
        ":test",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "gradient_input_output_exclusions",
    srcs = ["gradient_input_output_exclusions.py"],
    deps = [
        "//machina:machina_py",
        "//machina/python/autograph/pyct:anno",
        "//machina/python/autograph/pyct:cfg",
        "//machina/python/autograph/pyct:parser",
        "//machina/python/autograph/pyct:qual_names",
        "//machina/python/autograph/pyct:transformer",
        "//machina/python/autograph/pyct/static_analysis:activity",
        "//machina/python/autograph/pyct/static_analysis:liveness",
        "//machina/python/autograph/pyct/static_analysis:reaching_fndefs",
        "//machina/python/framework:op_def_registry",
        "//machina/python/framework:ops",
        "@pypi_gast//:pkg",
    ],
)

py_strict_binary(
    name = "gen_gradient_input_output_exclusions",
    srcs = ["gen_gradient_input_output_exclusions.py"],
    deps = [":gradient_input_output_exclusions"],
)

# Needed for the test below.
exports_files([
    "pywrap_gradient_exclusions.cc",
])

py_strict_test(
    name = "gradient_input_output_exclusions_test",
    srcs = ["gradient_input_output_exclusions_test.py"],
    data = [
        ":pywrap_gradient_exclusions.cc",
    ],
    tags = [
        "no_pip",  # No point linking the gen script in the pip package.
    ],
    deps = [
        ":gradient_input_output_exclusions",
        "//machina/python/lib/io:file_io",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:resource_loader",
    ],
)
