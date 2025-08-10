load("//machina:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("//machina:machina.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "op_callbacks_common",
    srcs = ["op_callbacks_common.py"],
)

py_strict_library(
    name = "check_numerics_callback",
    srcs = ["check_numerics_callback.py"],
    deps = [
        ":op_callbacks_common",
        ":source_utils",
        "//machina/core:protos_all_py",
        "//machina/python/eager:monitoring",
        "//machina/python/framework:op_callbacks",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:debug_ops_gen",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "dumping_callback",
    srcs = ["dumping_callback.py"],
    deps = [
        ":debug_events_writer",
        ":op_callbacks_common",
        ":source_utils",
        "//machina/core:protos_all_py",
        "//machina/python/eager:function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:op_callbacks",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:debug_ops_gen",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
        "//machina/python/util:object_identity",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_stack",
    ],
)

py_strict_library(
    name = "dumping_callback_test_lib",
    srcs = ["dumping_callback_test_lib.py"],
    deps = [
        ":check_numerics_callback",
        ":debug_events_reader",
        ":dumping_callback",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:versions",
    ],
)

py_strict_library(
    name = "common",
    srcs = ["common.py"],
)

py_strict_library(
    name = "debug_events_reader",
    srcs = ["debug_events_reader.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:errors",
        "//machina/python/framework:tensor_util",
        "//machina/python/lib/io:file_io",
        "//machina/python/lib/io:tf_record",
        "//machina/python/util:compat",
    ],
)

py_strict_library(
    name = "debug_events_monitors",
    srcs = ["debug_events_monitors.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "debug_events_writer",
    srcs = ["debug_events_writer.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/client:_pywrap_debug_events_writer",
    ],
)

py_strict_library(
    name = "debug_graphs",
    srcs = ["debug_graphs.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:op_def_registry",
        "//machina/python/platform:tf_logging",
    ],
)

py_strict_library(
    name = "debug_data",
    srcs = ["debug_data.py"],
    visibility = [
        "//machina:internal",
        "//third_party/py/tf_slim:__subpackages__",
        "//waymo/ml/deploy/numeric_debugging:__subpackages__",
    ],
    deps = [
        ":debug_graphs",
        "//machina/core:protos_all_py",
        "//machina/core/framework:graph_proto_py_proto",
        "//machina/core/util:event_proto_py_proto",
        "//machina/python/framework:tensor_util",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "debug_gradients",
    srcs = ["debug_gradients.py"],
    deps = [
        ":debug_data",
        ":debug_graphs",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/ops:array_ops_gen",
        "//machina/python/ops:variables",
    ],
)

py_strict_library(
    name = "debug_utils",
    srcs = ["debug_utils.py"],
)

py_strict_binary(
    name = "grpc_machina_server",
    srcs = ["grpc_machina_server.py"],
    deps = [
        ":grpc_machina_server_lib",
        "//machina/python/training:server_lib",
    ],
)

py_strict_library(
    name = "grpc_machina_server_lib",
    srcs = [
        "grpc_machina_server.py",
    ],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:server_lib",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "source_utils",
    srcs = ["source_utils.py"],
    deps = [
        ":profiling",
        "//third_party/py/numpy",
        "@absl_py//absl:app",
    ],
)

py_strict_library(
    name = "source_remote",
    srcs = ["source_remote.py"],
    deps = [
        ":common",
        ":debug_service_pb2_grpc",
        ":source_utils",
        "//machina/core:protos_all_py",
        "//machina/core/debug:debug_service_proto_py",
        "//machina/python/platform:gfile",
        "//machina/python/profiler:tfprof_logger",
    ],
)

py_strict_library(
    name = "profiling",
    srcs = ["profiling.py"],
)

py_strict_test(
    name = "common_test",
    size = "small",
    srcs = ["common_test.py"],
    deps = [
        ":common",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
    ],
)

py_strict_test(
    name = "debug_events_monitors_test",
    size = "medium",
    srcs = ["debug_events_monitors_test.py"],
    tags = [
        "no_windows",  # b/142475891
    ],
    deps = [
        ":debug_events_monitors",
        ":debug_events_reader",
        ":dumping_callback",
        ":dumping_callback_test_lib",
        "//machina/core:protos_all_py",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "debug_events_writer_test",
    size = "medium",
    srcs = ["debug_events_writer_test.py"],
    tags = [
        "no_windows",  # b/142475891
    ],
    deps = [
        ":debug_events_reader",
        ":debug_events_writer",
        ":dumping_callback_test_lib",
        "//machina/core:protos_all_py",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:versions",
        "//machina/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "debug_graphs_test",
    size = "small",
    srcs = ["debug_graphs_test.py"],
    deps = [
        ":debug_graphs",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_test(
    name = "debug_data_test",
    size = "small",
    srcs = ["debug_data_test.py"],
    tags = ["no_windows"],  # TODO(b/184424727): Enable this test on Windows.
    deps = [
        ":debug_data",
        "//machina/core:protos_all_py",
        "//machina/core/framework:graph_proto_py_proto",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "check_numerics_callback_test",
    size = "medium",
    srcs = ["check_numerics_callback_test.py"],
    tags = [
        "no_mac",  # TODO(b/175322370): Detected Infinity or NaN in output 0 of graph op "RealDiv"
        "no_windows",
    ],
    deps = [
        ":check_numerics_callback",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_grad",
        "//machina/python/ops:custom_gradient",
        "//machina/python/ops:gradient_checker_v2",
        "//machina/python/ops:math_grad",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops_gen",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "dumping_callback_test",
    size = "medium",
    srcs = ["dumping_callback_test.py"],
    shard_count = 4,
    tags = [
        "no_windows",  # TODO(b/142475891): Enable this test on Windows.
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":debug_events_reader",
        ":dumping_callback",
        ":dumping_callback_test_lib",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//machina/python/platform:tf_logging",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "debug_v2_ops_test",
    size = "medium",
    srcs = ["debug_v2_ops_test.py"],
    tags = ["no_windows_gpu"],
    deps = [
        ":debug_events_reader",
        ":debug_events_writer",
        ":dumping_callback_test_lib",
        "//machina/core:protos_all_py",
        "//machina/python/eager:def_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:debug_ops_gen",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

cuda_py_strict_test(
    name = "debug_gradients_test",
    size = "small",
    srcs = ["debug_gradients_test.py"],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":debug_data",
        ":debug_gradients",
        ":debug_utils",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:test",
        "//machina/python/training:gradient_descent",
    ],
)

py_strict_test(
    name = "debug_utils_test",
    size = "small",
    srcs = ["debug_utils_test.py"],
    deps = [
        ":debug_utils",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:test",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "source_utils_test",
    size = "small",
    srcs = ["source_utils_test.py"],
    tags = [
        "no_windows",
    ],
    deps = [
        ":debug_data",
        ":debug_utils",
        ":source_utils",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:test",
        "//machina/python/util:tf_inspect",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "source_remote_test",
    size = "small",
    srcs = ["source_remote_test.py"],
    tags = [
        "no_windows",
        "oss_serial",
    ],
    deps = [
        ":grpc_debug_test_server",
        ":source_remote",
        ":source_utils",
        "//machina/core/debug:debug_service_proto_py",
        "//machina/python/client:session",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//machina/python/util:tf_inspect",
    ],
)

py_strict_test(
    name = "profiling_test",
    size = "small",
    srcs = ["profiling_test.py"],
    deps = [
        ":profiling",
        "//machina/core:protos_all_py",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
    ],
)

py_strict_library(
    name = "session_debug_testlib",
    srcs = ["session_debug_testlib.py"],
    deps = [
        ":debug_data",
        ":debug_graphs",
        ":debug_utils",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:rnn",
        "//machina/python/ops:rnn_cell_impl",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:tensor_array_grad",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//machina/python/training:gradient_descent",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "debug_service_pb2_grpc",
    srcs = ["debug_service_pb2_grpc.py"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/core/debug:debug_service_proto_py",
    ],
)

py_strict_library(
    name = "grpc_debug_server",
    srcs = ["grpc_debug_server.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":debug_graphs",
        ":debug_service_pb2_grpc",
        "//machina/core:protos_all_py",
        "//machina/core/debug:debug_service_proto_py",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:compat",
    ],
)

py_strict_library(
    name = "grpc_debug_test_server",
    srcs = ["grpc_debug_test_server.py"],
    deps = [
        ":debug_data",
        ":debug_utils",
        ":grpc_debug_server",
        "//machina/core:protos_all_py",
        "//machina/core/debug:debug_service_proto_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:errors",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:variables",
        "//machina/python/util:compat",
        "@pypi_portpicker//:pkg",
    ],
)

cuda_py_strict_test(
    name = "debug_grappler_test",
    size = "small",
    srcs = ["debug_grappler_test.py"],
    xla_enable_strict_auto_jit = False,  # Tests TF:Classic implementation.
    deps = [
        ":debug_data",
        ":debug_utils",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/platform:test",
    ],
)

cuda_py_strict_test(
    name = "session_debug_file_test",
    size = "small",
    srcs = ["session_debug_file_test.py"],
    tags = ["notsan"],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":debug_data",
        ":debug_utils",
        ":session_debug_testlib",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/platform:test",
    ],
)

cuda_py_strict_test(
    name = "debug_graph_reconstruction_test",
    size = "small",
    srcs = ["debug_graph_reconstruction_test.py"],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":debug_data",
        ":debug_graphs",
        ":debug_utils",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:cond",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:gradient_descent",
    ],
)

cuda_py_strict_test(
    name = "session_debug_multi_gpu_test",
    size = "small",
    srcs = ["session_debug_multi_gpu_test.py"],
    tags = [
        "no_windows",  # TODO(b/184424727): Re-enable this.
        "no_windows_gpu",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":debug_data",
        ":debug_utils",
        "//machina/core:protos_all_py",
        "//machina/python/client:device_lib",
        "//machina/python/client:session",
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:file_io",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:test",
    ],
)
