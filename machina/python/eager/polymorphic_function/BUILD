load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")
load("//machina/compiler/tests:build_defs.bzl", "tf_xla_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "attributes",
    srcs = ["attributes.py"],
    visibility = ["//machina/python:__subpackages__"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/util:compat",
    ],
)

py_strict_library(
    name = "autograph_util",
    srcs = ["autograph_util.py"],
    deps = [
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/impl:api",
        "//machina/python/util:tf_decorator_py",
    ],
)

py_strict_library(
    name = "transform",
    srcs = ["transform.py"],
    visibility = ["//machina/python:__subpackages__"],
    deps = [],
)

pytype_strict_library(
    name = "atomic_function",
    srcs = ["atomic_function.py"],
    visibility = ["//machina/python:__subpackages__"],
    deps = [
        ":attributes",
        ":function_type_utils",
        "//machina/core:protos_all_py",
        "//machina/core/function/polymorphism:function_type",
        "//machina/python/client:pywrap_tf_session",
        "//machina/python/eager:context",
        "//machina/python/eager:record",
        "//machina/python/framework:auto_control_deps_utils",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:error_interpolation",
        "//machina/python/framework:errors",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:function_def_to_graph",
        "//machina/python/framework:ops",
        "//machina/python/ops:handle_data_util",
        "//machina/python/types:core",
        "//machina/python/util:compat",
        "//machina/python/util:function_utils",
        "//machina/python/util:tf_stack",
    ],
)

cuda_py_strict_test(
    name = "atomic_function_test",
    size = "medium",
    srcs = ["atomic_function_test.py"],
    deps = [
        ":atomic_function",
        ":polymorphic_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "concrete_function",
    srcs = ["concrete_function.py"],
    visibility = ["//machina/python:__subpackages__"],
    deps = [
        ":atomic_function",
        ":attributes",
        ":function_type_utils",
        ":saved_model_exported_concrete",
        "//machina/core:protos_all_py",
        "//machina/core/function/polymorphism:function_type",
        "//machina/python:pywrap_tfe",
        "//machina/python/eager:backprop_util",
        "//machina/python/eager:context",
        "//machina/python/eager:forwardprop_util",
        "//machina/python/eager:graph_only_ops",
        "//machina/python/eager:record",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:type_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:default_gradient",
        "//machina/python/ops:gradients_util",
        "//machina/python/ops:handle_data_util",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/profiler:trace",
        "//machina/python/trackable:base",
        "//machina/python/types:core",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//machina/python/util:object_identity",
    ],
)

cuda_py_strict_test(
    name = "concrete_function_test",
    size = "small",
    srcs = ["concrete_function_test.py"],
    deps = [
        ":atomic_function",
        ":concrete_function",
        ":polymorphic_function",
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "tracing_compilation",
    srcs = ["tracing_compilation.py"],
    visibility = ["//machina/python:__subpackages__"],
    deps = [
        ":attributes",
        ":concrete_function",
        ":function_context",
        ":function_type_utils",
        ":tf_method_target",
        ":transform",
        "//machina/core/function/capture:capture_container",
        "//machina/core/function/polymorphism:function_cache",
        "//machina/core/function/polymorphism:function_type",
        "//machina/core/function/trace_type",
        "//machina/python/autograph/core:ag_ctx",
        "//machina/python/eager:monitoring",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/profiler:trace",
        "//machina/python/util:compat",
    ],
)

py_strict_library(
    name = "tf_method_target",
    srcs = ["tf_method_target.py"],
    visibility = [
        "//machina/python/autograph/impl:__pkg__",
        "//machina/python/eager:__pkg__",
    ],
    deps = [
        "//machina/python/util:tf_inspect",
    ],
)

py_strict_library(
    name = "polymorphic_function",
    srcs = ["polymorphic_function.py"],
    visibility = ["//machina:internal"],
    deps = [
        ":attributes",
        ":autograph_util",
        ":compiler_ir",
        ":eager_function_run",
        ":function_type_utils",
        ":tf_method_target",
        ":tracing_compilation",
        "//machina/core:protos_all_py",
        "//machina/core/function/capture:capture_container",
        "//machina/core/function/polymorphism:function_cache",
        "//machina/core/function/trace_type",
        "//machina/python/distribute/parallel_device",
        "//machina/python/eager:context",
        "//machina/python/eager:lift_to_graph",
        "//machina/python/eager:monitoring",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:errors",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:control_flow_util",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/profiler:trace",
        "//machina/python/trackable:base",
        "//machina/python/types:core",
        "//machina/python/util:deprecation",
        "//machina/python/util:nest",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
        "//machina/python/util:traceback_utils",
    ],
)

cuda_py_strict_test(
    name = "polymorphic_function_test",
    size = "medium",
    srcs = ["polymorphic_function_test.py"],
    shard_count = 15,
    tags = [
        "nomac",  # b/157056289
    ],
    # TODO(b/169371527): insert transfer op in eager lowering for TFRT.
    deps = [
        ":attributes",
        ":polymorphic_function",
        "//machina/core/function/capture:capture_container",
        "//machina/core/function/trace_type",
        "//machina/python/autograph/core:ag_ctx",
        "//machina/python/autograph/core:converter",
        "//machina/python/autograph/lang:directives",
        "//machina/python/checkpoint",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:iterator_ops",
        "//machina/python/eager:backprop",
        "//machina/python/eager:cancellation",
        "//machina/python/eager:context",
        "//machina/python/eager:lift_to_graph",
        "//machina/python/framework:composite_tensor",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:extension_type",
        "//machina/python/framework:function",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:random_seed",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:test_ops",
        "//machina/python/framework:type_spec",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:clip_ops",
        "//machina/python/ops:cond_v2",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:functional_ops",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:list_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:random_ops_gen",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:script_ops",
        "//machina/python/ops:sendrecv_ops_gen",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:training_ops_gen",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/ops/structured:structured_tensor",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:save",
        "//machina/python/saved_model:save_context",
        "//machina/python/saved_model:save_options",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "//machina/python/util:tf_decorator_py",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "polymorphic_function_test_cpu_only",
    srcs = ["polymorphic_function_test_cpu_only.py"],
    # --config=cuda implicitly links in XLA.
    tags = [
        "no_cuda_on_cpu_tap",
        "no_gpu",
        "no_oss",  # No way to force no XLA linkage in OSS build from here.
        "no_pip",
    ],
    deps = [
        ":polymorphic_function",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_xla_py_strict_test(
    name = "polymorphic_function_xla_jit_test",
    srcs = ["polymorphic_function_xla_jit_test.py"],
    # copybara:uncomment_begin
    # #TODO(b/185944215) # Remove once the bug is fixed.
    # disable_tpu_tfrt = True,
    # copybara:uncomment_end
    disabled_backends = [
        "cpu_ondemand",
    ],
    enable_mlir_bridge = True,
    tags = [
        "no_mac",
        "no_oss",  # TODO(b/295654746)
        "no_pip",
        "no_tfrt",  # TODO(b/185944215)
        "no_windows",
    ],
    use_xla_device = False,
    deps = [
        ":polymorphic_function",
        "//machina/compiler/tests:xla_test",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:collective_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_assert",
        "//machina/python/ops:control_flow_util",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:summary_ops_v2",
        "//machina/python/ops:tensor_array_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:nest",
    ],
)

tf_xla_py_strict_test(
    name = "polymorphic_function_xla_test",
    srcs = ["polymorphic_function_xla_test.py"],
    # copybara:uncomment_begin
    # #TODO(b/185944215) # Remove once the bug is fixed.
    # disable_tpu_tfrt = True,
    # copybara:uncomment_end
    enable_mlir_bridge = False,
    tags = [
        "no_pip",
        "no_windows",
        "nomac",
    ],
    deps = [
        ":polymorphic_function",
        "//machina/compiler/tests:xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "gradients_test",
    size = "medium",
    srcs = ["gradients_test.py"],
    shard_count = 5,
    deps = [
        ":polymorphic_function",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_grad",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "composite_tensor_utils",
    srcs = ["composite_tensor_utils.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python/framework:composite_tensor",
        "//machina/python/util:_pywrap_utils",
        "//machina/python/util:nest",
    ],
)

tf_py_strict_test(
    name = "tracing_compilation_test",
    size = "medium",
    srcs = ["tracing_compilation_test.py"],
    deps = [
        ":function_type_utils",
        ":tracing_compilation",
        "//machina/core:protos_all_py",
        "//machina/core/function/capture:capture_container",
        "//machina/core/function/polymorphism:function_cache",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:test_ops",
        "//machina/python/module",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:functional_ops_gen",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:resource_variable_ops_gen",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/ops/ragged:ragged_factory_ops",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "//machina/python/saved_model:load",
        "//machina/python/saved_model:save",
        "//machina/python/util:compat",
        "//machina/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "argument_naming_test",
    size = "medium",
    srcs = ["argument_naming_test.py"],
    # TODO(b/169371527): insert transfer op in eager lowering for TFRT.
    deps = [
        ":polymorphic_function",
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "collection_test",
    size = "medium",
    srcs = ["collection_test.py"],
    deps = [
        ":polymorphic_function",
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "eager_function_run",
    srcs = ["eager_function_run.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "function_context",
    srcs = ["function_context.py"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/core/function/polymorphism:function_cache",
        "//machina/python/eager:context",
        "//machina/python/framework:device",
        "//machina/python/framework:func_graph",
        "//machina/python/framework:ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/saved_model:save_context",
    ],
)

py_strict_library(
    name = "saved_model_utils",
    srcs = ["saved_model_utils.py"],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/saved_model/registration",
        "//machina/python/trackable:base",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "saved_model_exported_concrete",
    srcs = ["saved_model_exported_concrete.py"],
    visibility = ["//machina:internal"],
    deps = [
        ":function_type_utils",
        "//machina/python/trackable:base",
    ],
)

py_strict_library(
    name = "function_type_utils",
    srcs = ["function_type_utils.py"],
    visibility = ["//machina:internal"],
    deps = [
        ":composite_tensor_utils",
        "//machina/core/function/polymorphism:function_type",
        "//machina/core/function/trace_type",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:type_spec",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/util:nest",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

tf_py_strict_test(
    name = "function_spec_test",
    size = "medium",
    srcs = ["function_spec_test.py"],
    deps = [
        ":function_type_utils",
        "//machina/core/function/polymorphism:function_type",
        "//machina/core/function/trace_type",
        "//machina/python/framework:tensor_spec",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:tf_decorator_py",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "compiler_ir",
    srcs = ["compiler_ir.py"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/core/function/trace_type",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor_spec",
        "//machina/python/ops:random_ops",
        "//machina/python/util:nest",
    ],
)

tf_xla_py_strict_test(
    name = "compiler_ir_test",
    srcs = ["compiler_ir_test.py"],
    disabled_backends = [
        "cpu_ondemand",
        "gpu_a100",
        "gpu_h100",
    ],
    enable_mlir_bridge = True,
    tags = [
        "no_mac",
        "no_pip",
        "no_tfrt",  # TODO(b/185944215)
        "no_windows",
    ],
    use_xla_device = False,
    deps = [
        ":compiler_ir",
        ":polymorphic_function",
        "//machina/compiler/tests:xla_test",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:nest",
    ],
)
