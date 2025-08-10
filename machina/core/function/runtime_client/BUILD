load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_test")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "tf_python_pybind_extension")
load("//machina/core/platform:build_config_root.bzl", "if_static")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/core/function:__subpackages__"],
)

licenses(["notice"])

cc_library(
    name = "runtime_client_cc",
    srcs = [
        "runtime_client.cc",
    ],
    hdrs = [
        "runtime_client.h",
    ],
    defines = select({
        "//machina/compiler/mlir/python:disable_mlir_config": ["DISABLE_MLIR"],
        "//conditions:default": [],
    }),
    visibility = ["//machina:__subpackages__"],
    deps = [
        "//machina/c/eager:abstract_tensor_handle",
        "//machina/c/eager:immediate_execution_context",
        "//machina/c/eager:immediate_execution_operation",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/compiler/mlir/machina:error_util",
        "//machina/compiler/mlir/machina:import_model",
        "//machina/compiler/mlir/machina:mlir_roundtrip_flags",
        "//machina/compiler/mlir/tf2xla/api/v2:graph_to_tf_executor",
        "//machina/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//machina/core:core_cpu",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core/common_runtime:function_def_utils",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/common_runtime/eager:core",
        "//machina/core/framework:function_proto_cc",
        "//machina/core/framework:graph_proto_cc",
        "//machina/core/framework:op_def_proto_cc",
        "//machina/core/ir:Dialect",
        "//machina/core/ir/importexport:graphdef_export",
        "//machina/core/ir/importexport:graphdef_import",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "//machina/core/platform:stringpiece",
        "//machina/core/platform:types",
        "//machina/core/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
    ] + select({
        "//machina/compiler/mlir/python:disable_mlir_config": [],
        "//conditions:default": [
            "//machina/compiler/mlir/python:mlir",
        ],
    }),
    # TODO(mdan): Get rid of alwayslink, it's nonstandard.
    alwayslink = 1,
)

# TODO(mdan): Pull these transitive header deps in a more decent fashion.
# TODO(mdan): Get rid of headers-only lib, it's nonstandard. Use cc_shared_library?
cc_library(
    name = "runtime_client_headers",
    textual_hdrs = [
        "runtime_client.h",
        "//machina/c/eager:pywrap_required_hdrs",
        "//machina/core/common_runtime/eager:pywrap_required_hdrs",
        "//machina/core/config:flags_headers",
        "//machina/core/distributed_runtime:pywrap_required_hdrs",
        "//machina/core/distributed_runtime/coordination:pywrap_required_hdrs",
        "//machina/core/distributed_runtime/eager:pywrap_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime:pywrap_required_hdrs",
        "@local_xla//xla/tsl/distributed_runtime/coordination:pywrap_required_hdrs",
    ],
)

tf_cc_test(
    name = "runtime_client_cc_test",
    srcs = ["runtime_client_test.cc"],
    deps = [
        ":runtime_client_cc",
        "//machina/c:c_api_experimental",  # buildcleaner: keep (registers CPU ops?)
        "//machina/c:tensor_interface",
        "//machina/c/eager:immediate_execution_tensor_handle",
        "//machina/core:test",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/framework:function_proto_cc",
        "//machina/core/framework:op_def_proto_cc",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/function/testing:test_pass_cc",
        "//machina/core/ir:Dialect",
        "//machina/core/platform:protobuf",
        "//machina/core/platform:status",
        "//machina/core/platform:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:Parser",
    ],
)

# TODO(b/221223841): Get rid of if_static, it's nonstandard.
tf_python_pybind_extension(
    name = "runtime_client_pybind",
    srcs = ["runtime_client_pybind.cc"],
    data = [
        "runtime_client_pybind.pyi",
    ],
    enable_stub_generation = True,
    deps = [
        ":runtime_client_headers",
        "//machina/python/lib/core:pybind11_status",
        "@com_google_absl//absl/status",
        "@pybind11",
    ] + if_static(
        extra_deps = [
            "//machina/core/framework:function_proto_cc",
            "//machina/core/protobuf:eager_service_proto_cc",
            "//machina/core/protobuf:master_proto_cc",
            "//machina/core/protobuf:worker_proto_cc",
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc",
        ],
        otherwise = [
            "//machina/core/framework:function_proto_cc_headers_only",
            "//machina/core/protobuf:eager_service_proto_cc_headers_only",
            "//machina/core/protobuf:master_proto_cc_headers_only",
            "//machina/core/protobuf:worker_proto_cc_headers_only",
            "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc_headers_only",
        ],
    ),
)

# TODO(mdan): Drop function_proto_py_pb2 once pybind11_protobuf is available.
pytype_strict_library(
    name = "runtime_client_py",
    srcs = [
        "runtime_client.py",
    ],
    visibility = [
        "//learning/brain/experimental/tfq:__subpackages__",
        "//machina/core/function/transform:__subpackages__",
        "//machina/python/eager:__subpackages__",
    ],
    deps = [
        ":runtime_client_pybind",
        "//machina/core/framework:function_proto_py",
        "//machina/python:pywrap_machina",  # buildcleaner: keep (required for TF pybind)
    ],
)

py_strict_test(
    name = "runtime_client_py_test",
    srcs = ["runtime_client_test.py"],
    main = "runtime_client_test.py",
    tags = ["no_oss"],  # TODO(b/219089812)
    deps = [
        ":runtime_client_py",
        #internal proto upb dep
        "//machina/core/framework:function_proto_py",
        "//machina/core/function/testing:test_pass_py",
        "//machina/python:tf2",
        "//machina/python/eager:context",
        "//machina/python/eager:def_function",
        "//machina/python/eager:execute",
        "//machina/python/eager:remote",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:math_ops",
        "//machina/python/platform:client_testlib",
    ],
)
