load("@rules_cc//cc:cc_library.bzl", "cc_library")

# Contains targets that expose client session APIs
load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.bzl", "tf_cuda_library")
load("//machina:machina.default.bzl", "cuda_py_strict_test", "tf_py_strict_test", "tf_python_pybind_extension")
load("//machina/core/platform:build_config_root.bzl", "if_pywrap", "if_static")
load(
    "//machina/tools/test:performance.bzl",
    "cuda_py_benchmark_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:internal",
    ],
    licenses = ["notice"],
)

py_strict_library(
    name = "pywrap_tf_session",
    srcs = ["pywrap_tf_session.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":_pywrap_tf_session",
        "//machina/python:pywrap_machina",
        "//machina/python/util:tf_stack",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_tf_session",
    srcs = ["tf_session_wrapper.cc"],
    hdrs = if_pywrap(
        if_false = [
            "tf_session_helper.h",
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
            "//machina/python/lib/core:safe_pyobject_ptr_required_hdrs",
        ],
        if_true = [],
    ),
    additional_exported_symbols = [
        "_TF_SetTarget",
        "_TF_SetConfig",
        "_TF_NewSessionOptions",
    ],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_tf_session.pyi",
    ],
    deps = [
        "//machina/c:pywrap_required_hdrs",
        "//machina/core:framework_headers_lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_headers_lib",
        "//machina/core/config:flags_headers",
        "//machina/core/framework:pywrap_required_hdrs",
        "//machina/core/lib/llvm_rtti",
        "//machina/core/public:release_version",
        "//machina/core/util:version_info",
        "//machina/python/lib/core:pybind11_lib",
        "//machina/python/lib/core:pybind11_status",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "//third_party/py/numpy:headers",
        "@com_google_absl//absl/types:optional",
        "@eigen_archive//:eigen3",
        "@local_xla//third_party/python_runtime:headers",
        "@local_xla//xla/tsl/python/lib/core:numpy",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ] + if_pywrap([
        "//machina/c:safe_ptr",
        "//machina/c:c_api_experimental",
        "//machina/python/client:tf_session_helper",
        "//machina/c:python_api",
        "//machina/core/common_runtime:core_cpu_lib",
    ]) + if_static(
        extra_deps = [
            "//machina/core/protobuf:eager_service_proto_cc",
            "//machina/core/protobuf:master_proto_cc",
            "//machina/core/protobuf:worker_proto_cc",
            "//machina/core:version_lib",
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

tf_python_pybind_extension(
    name = "_pywrap_debug_events_writer",
    srcs = ["debug_events_writer_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_debug_events_writer.pyi",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core:framework_headers_lib",
        "//machina/core:lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/python/lib/core:pybind11_absl",
        "//machina/python/lib/core:pybind11_proto",
        "//machina/python/lib/core:pybind11_status",
        "@com_google_absl//absl/strings",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_events_writer",
    srcs = ["events_writer_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_events_writer.pyi",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core:framework_headers_lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/python/lib/core:pybind11_absl",
        "//machina/python/lib/core:pybind11_proto",
        "//machina/python/lib/core:pybind11_status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

py_strict_library(
    name = "client",
    srcs = ["client_lib.py"],
    deps = [
        ":_pywrap_device_lib",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_machina",
        "//machina/python/client:session",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/platform:build_info",
        "//machina/python/platform:tf_logging",
    ],
)

tf_py_strict_test(
    name = "events_writer_test",
    size = "small",
    srcs = ["events_writer_test.py"],
    deps = [
        ":_pywrap_events_writer",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_machina",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/lib/io:tf_record",
        "//machina/python/platform:test",
        "//machina/python/util:compat",
    ],
)

py_strict_library(
    name = "device_lib",
    srcs = ["device_lib.py"],
    visibility = [
        "//machina:internal",
        "//third_party/mlperf/submissions/training/v0_7/models:__subpackages__",
        "//third_party/py/cleverhans:__subpackages__",
    ],
    deps = [
        ":_pywrap_device_lib",
        "//machina/core:protos_all_py",
        "//machina/python:pywrap_machina",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_device_lib",
    srcs = ["device_lib_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_device_lib.pyi",
    ],
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:framework_internal_headers_lib",
        "//machina/core:lib_proto_parsing",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime:core_cpu_headers_lib",
        "//machina/core/common_runtime:device",
        "//machina/core/common_runtime:device_factory",
        "//machina/python/lib/core:pybind11_proto",
        "//machina/python/lib/core:pybind11_status",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

cuda_py_strict_test(
    name = "device_lib_test",
    size = "small",
    srcs = [
        "device_lib_test.py",
    ],
    deps = [
        ":device_lib",
        "//machina/core:protos_all_py",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
    ],
)

cc_library(
    name = "session_ref",
    srcs = ["session_ref.cc"],
    hdrs = ["session_ref.h"],
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/protobuf:replay_log_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
    ] + if_static(
        extra_deps = [
            "//machina/core/protobuf:master_proto_cc",
        ],
        otherwise = [
            "//machina/core/protobuf:master_proto_cc_headers_only",
        ],
    ),
)

tf_cuda_library(
    name = "tf_session_helper",
    srcs = ["tf_session_helper.cc"],
    hdrs = ["tf_session_helper.h"],
    compatible_with = [],
    deps = [
        ":construction_fails_op",
        ":session_ref",
        "//machina/c:c_api",
        "//machina/c:c_api_internal",
        "//machina/c:safe_ptr",
        "//machina/c:tf_buffer_internal",
        "//machina/c:tf_status_helper",
        "//machina/core",
        "//machina/core:all_kernels",
        "//machina/core:core_cpu_base",
        "//machina/core:direct_session",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/python/framework:test_ops_kernels",
        "//machina/python/lib/core:ndarray_tensor",
        "//machina/python/lib/core:ndarray_tensor_bridge",
        "//machina/python/lib/core:safe_pyobject_ptr",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/python/lib/core:numpy",
    ],
    alwayslink = 1,
)

py_strict_library(
    name = "session",
    srcs = ["session.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = [
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
        ":pywrap_tf_session",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/eager:monitoring",
        "//machina/python/framework:device",
        "//machina/python/framework:error_interpolation",
        "//machina/python/framework:errors",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:stack",
        "//machina/python/framework:tensor",
        "//machina/python/ops:session_ops",
        "//machina/python/platform:tf_logging",
        "//machina/python/training/experimental:mixed_precision_global_state",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:nest",
        "//machina/python/util:numpy_compat",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
        "@pypi_wrapt//:pkg",
    ],
)

py_strict_library(
    name = "timeline",
    srcs = ["timeline.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/platform:build_info",
        "//machina/python/platform:tf_logging",
    ],
)

# Just used by tests.
tf_cuda_library(
    name = "construction_fails_op",
    srcs = ["test_construction_fails_op.cc"],
    deps = [
        "//machina/core",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_py_strict_test(
    name = "session_test",
    size = "medium",
    srcs = ["session_test.py"],
    grpc_enabled = True,
    tags = [
        "no_gpu",  # b/127001953
        "no_oss",  # b/198485096
        "no_pip_gpu",  # testInteractivePlacePrunedGraph fails on invalid assumption about GPU ops.
        "no_windows",
    ],
    deps = [
        ":session",
        "//machina/core:protos_all_py",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/framework:config",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:device",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:function",
        "//machina/python/framework:importer",
        "//machina/python/framework:indexed_slices",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:stack",
        "//machina/python/framework:tensor_util",
        "//machina/python/framework:test_lib",
        "//machina/python/framework:versions",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:control_flow_ops",
        "//machina/python/ops:control_flow_ops_gen",
        "//machina/python/ops:data_flow_ops",
        "//machina/python/ops:gradients",
        "//machina/python/ops:gradients_impl",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variable_v1",
        "//machina/python/ops:variables",
        "//machina/python/ops:while_loop",
        "//machina/python/platform:test",
        "//machina/python/training:server_lib",
        "//machina/python/util:compat",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

tf_py_strict_test(
    name = "session_clusterspec_prop_test",
    size = "small",
    srcs = ["session_clusterspec_prop_test.py"],
    grpc_enabled = True,
    tags = [
        "no_gpu",
        "no_oss",
        "no_pip_gpu",
        "notap",
    ],
    deps = [
        ":session",
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:state_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:test",
        "//machina/python/training:server_lib",
        "//third_party/py/numpy",
    ],
)

tf_py_strict_test(
    name = "session_list_devices_test",
    size = "small",
    srcs = ["session_list_devices_test.py"],
    grpc_enabled = True,
    tags = [
        "no_gpu",
        "no_pip_gpu",
    ],
    deps = [
        ":pywrap_tf_session",
        ":session",
        "//machina/core:protos_all_py",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/platform:test",
        "//machina/python/training:server_lib",
    ],
)

tf_py_strict_test(
    name = "session_partial_run_test",
    size = "small",
    srcs = ["session_partial_run_test.py"],
    grpc_enabled = True,
    tags = [
        "no_gpu",
        "no_rocm",
        "no_windows",
    ],
    deps = [
        ":session",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:test",
        "//machina/python/training:server_lib",
    ],
)

cuda_py_strict_test(
    name = "timeline_test",
    size = "small",
    srcs = ["timeline_test.py"],
    tags = [
        "gpu_cupti",
        "no_gpu",  # b/154742661
    ],
    xla_enable_strict_auto_jit = False,  # Graph structure is different with autojit
    deps = [
        ":session",
        ":timeline",
        "//machina/core:protos_all_py",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
    ],
)

cuda_py_strict_test(
    name = "virtual_gpu_test",
    size = "small",
    srcs = ["virtual_gpu_test.py"],
    tags = [
        "no_gpu",  # b/127386241
        "no_windows_gpu",
    ],
    deps = [
        "//machina/core:protos_all_py",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:tf_logging",
    ],
)

cuda_py_benchmark_test(
    name = "session_benchmark",
    srcs = ["session_benchmark.py"],
    grpc_enabled = True,
    main = "session_benchmark.py",
    deps = [
        ":session",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:random_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//machina/python/training:server_lib",
        "//third_party/py/numpy",
    ],
)
