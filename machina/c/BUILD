# Description:
# C API for TensorFlow, for use by client language bindings.

load("@bazel_skylib//lib:selects.bzl", "selects")
load("@local_config_tensorrt//:build_defs.bzl", "if_tensorrt")
load("@local_xla//xla/tsl/platform:build_config_root.bzl", "if_pywrap")
load("@local_xla//xla/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "check_deps",
    "if_google",
    "if_not_mobile",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_library",
    "tf_custom_op_library",
    "tf_kernel_library",
)
load("//machina:machina.default.bzl", "filegroup", "tf_cuda_cc_test")
load(
    "//machina/core/tpu:build_defs.bzl",
    "if_libtpu_tf_status",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "safe_ptr_hdr",
    srcs = ["safe_ptr.h"],
    visibility = [
        "//machina:internal",
    ],
)

cc_library(
    name = "safe_ptr",
    srcs = [
        "safe_ptr.cc",
        "//machina/c/eager:headers",
    ],
    hdrs = ["safe_ptr.h"],
    visibility = [
        "//machina:internal",
    ],
    deps = [
        ":c_api_internal",
    ],
)

# -----------------------------------------------------------------------------
# Public targets

filegroup(
    name = "headers",
    srcs = [
        "c_api.h",
        "c_api_experimental.h",
        "c_api_macros.h",
        "tensor_interface.h",
        "tf_attrtype.h",
        "tf_buffer.h",
        "tf_datatype.h",
        "tf_file_statistics.h",
        "tf_status.h",
        "tf_tensor.h",
        "tf_tensor_helper.h",
        "tf_tstring.h",
        "//machina/core/platform:ctstring",
        "@local_xla//xla/tsl/c:headers",
    ] + if_tensorrt([
        "//machina/compiler/tf2tensorrt:headers",
    ]),
    visibility = ["//machina:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "c_api_experimental.cc",
            "c_api_experimental.h",
            "python_api.cc",
            "python_api.h",
            "*test*",
        ],
    ) + [
        "//machina/cc:srcs_no_runtime",
        "//machina/core/distributed_runtime:server_lib.h",
        "@local_tsl//tsl/platform:ctstring",
        "@local_xla//xla/tsl/c:srcs",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pywrap_required_hdrs",
    textual_hdrs = [
        "c_api_internal.h",
        "c_api_macros.h",
        "conversion_macros.h",
        "python_api.h",
        "tensor_interface.h",
        "tf_status_helper.h",
        "tf_buffer_internal.h",
        "tf_status_internal.h",
        "tf_tensor_helper.h",
        "tf_tensor_internal.h",
        "@local_xla//xla/tsl/c:tsl_status_internal_headers",
    ],
    visibility = [
        "//machina/core:__pkg__",
        "//machina/python:__subpackages__",
    ],
)

cc_library(
    name = "c_api_headers",
    hdrs = [
        "c_api.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        ":tf_attrtype",
        ":tf_buffer_hdrs",
        ":tf_datatype_hdrs",
        ":tf_status_headers",
        ":tf_tensor_hdrs",
        # TODO: Only include tf_tstring_hdrs. Don't expose the implementation of TF_TString to API
        # users.
        ":tf_tstring",
        "//machina/core:protos_all_cc",
    ],
)

tf_cuda_library(
    name = "c_api_internal",
    hdrs = [
        "c_api.h",
        "c_api_internal.h",
        "c_api_macros.h",
        "tf_buffer.h",
        "tf_datatype.h",
        "tf_tensor.h",
        "tf_tstring.h",
    ],
    visibility = ["//visibility:public"],
    deps = selects.with_or({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        ("//machina:chromiumos", "//machina:fuchsia"): [
            ":tf_attrtype",
            "//machina/core:core_cpu",
            "//machina/core:framework",
            "//machina/core:lib",
            "//machina/core/platform:platform",
        ],
        "//conditions:default": [
            ":tf_attrtype",
            "//machina/core:core_cpu",
            "//machina/core:framework",
            "//machina/core:lib",
            "//machina/core/platform:platform",
            "//machina/core:op_gen_lib",
            "//machina/core/distributed_runtime:server_lib",
        ],
    }) + [
        ":tf_buffer_internal",
        ":tf_status_helper",
        ":tf_status_internal",
        ":tf_tensor_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/status",
    ],
)

filegroup(
    name = "pywrap_tf_session_hdrs",
    srcs = [
        "python_api.h",
    ],
    visibility = [
        "//machina/core:__pkg__",
        "//machina/python:__pkg__",
    ],
)

cc_library(
    name = "tf_attrtype",
    hdrs = ["tf_attrtype.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "c_api_macros_hdrs",
    hdrs = [
        "c_api_macros.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "c_api_macros",
    hdrs = [
        "c_api_macros.h",
        "c_api_macros_internal.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":tf_status",
        "//machina/core:lib",
        "//machina/core:lib_internal",
    ],
)

tf_cuda_library(
    name = "c_api",
    hdrs = [
        "c_api.h",
        "tf_attrtype.h",
        "tf_buffer.h",
        "tf_datatype.h",
        "tf_file_statistics.h",
        "tf_status.h",
        "tf_tensor.h",
        "tf_tstring.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_internal",
        ":c_api_macros_hdrs",
        ":c_api_no_xla",
        ":tf_attrtype",
        ":tf_buffer",
        ":tf_file_statistics",
        ":tf_status_helper",
        ":tf_status_internal",
        ":tf_tensor_internal",
        ":tf_tstring",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:tstring",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/c:tsl_status",
    ] + select({
        "//machina:with_xla_support": [
            "//machina/compiler/jit",
            "//machina/compiler/tf2xla:xla_compiler",
        ],
        "//conditions:default": [],
    }) + if_tensorrt([
        "//machina/compiler/tf2tensorrt:trt_convert_api",
    ]),
)

# Check that c_api_no_xla does not depend on xla.
check_deps(
    name = "c_api_no_xla_check_deps",
    disallowed_deps = ["//machina/compiler/jit:xla_kernel_creator"],
    deps = [":c_api_no_xla"],
)

tf_cuda_library(
    name = "c_api_no_xla",
    srcs = [
        "c_api.cc",
        "c_api_function.cc",
    ],
    hdrs = [
        "c_api.h",
    ],
    copts = tf_copts(),
    visibility = [
        "//machina:__subpackages__",
        "//machina/python:__subpackages__",
    ],
    deps = [
        ":c_api_internal",
        ":tf_attrtype",
        ":tf_buffer",
        ":tf_buffer_internal",
        ":tf_datatype",
        ":tf_status_helper",
        ":tf_status_internal",
        "//machina/core/public:release_version",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            ":env",
            ":logging",
            ":tf_status",
            ":tf_tensor",
            "//machina/c/experimental/filesystem:modular_filesystem",
            "//machina/cc:grad_ops",
            "//machina/cc:gradients",
            "//machina/cc:ops",
            "//machina/cc:scope_internal",
            "//machina/cc:while_loop",
            "//machina/cc/saved_model:loader_lite",
            "//machina/core:core_cpu",
            "//machina/core:core_cpu_internal",
            "//machina/core:framework",
            "//machina/core:lib",
            "//machina/core:lib_internal",
            "//machina/core:op_gen_lib",
            "//machina/core:protos_all_cc",
            "//machina/core/config:flag_defs",
            "//machina/core/config:flags",
            "//machina/core/distributed_runtime:server_lib",
            "//machina/core/kernels:logging_ops",
            "@com_google_absl//absl/strings",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "logging",
    srcs = ["logging.cc"],
    hdrs = ["logging.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros",
        "//machina/core/platform:logging",
        "//machina/core/platform:stringprintf",
    ],
)

tf_cuda_library(
    name = "tf_status_internal",
    hdrs = [
        "tf_status.h",
        "tf_status_internal.h",
        "@local_xla//xla/tsl/c:tsl_status_internal_headers",
    ],
    visibility = [
        "//machina/c:__subpackages__",
        # copybara:uncomment_begin(google-only)
        # "//machina/cc/experimental/tf2:__pkg__",
        # "//machina/cc/experimental/tf2:__subpackages__",
        # copybara:uncomment_end
        "//machina/compiler/mlir/machina/c:__subpackages__",
        "//machina/core/transforms:__subpackages__",
    ],
    deps = [
        ":c_api_macros_hdrs",
        "@local_xla//xla/tsl/c:tsl_status",
        "@local_xla//xla/tsl/c:tsl_status_internal",
        "@local_xla//xla/tsl/platform:status",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core:lib",
        ],
    }),
)

filegroup(
    name = "tf_status_internal_headers",
    srcs = [
        "tf_status_internal.h",
        "@local_xla//xla/tsl/c:tsl_status_internal_headers",
    ],
    visibility = [
        "//machina/python:__subpackages__",
    ],
)

cc_library(
    name = "tf_shape",
    srcs = ["tf_shape.cc"],
    hdrs = ["tf_shape.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros",
        ":tf_shape_internal",
        "//machina/core:framework",
    ],
)

cc_library(
    name = "tf_shape_internal",
    hdrs = ["tf_shape_internal.h"],
    copts = tf_copts(),
    visibility = ["//machina:internal"],
    deps = [
        ":conversion_macros",
        "//machina/core:framework",
    ],
)

cc_library(
    name = "tf_status",
    srcs = ["tf_status.cc"],
    hdrs = ["tf_status.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        ":tf_status_internal",
        "@local_xla//xla/tsl/c:tsl_status",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core:lib",
        ],
    }),
)

cc_library(
    name = "tf_status_headers",
    hdrs = ["tf_status.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        "@local_xla//xla/tsl/c:tsl_status_headers",
    ],
)

cc_library(
    name = "tf_tstring",
    srcs = [
        "tf_tstring.cc",
    ],
    hdrs = [
        "tf_tstring.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        ":tf_datatype_hdrs",
        ":tf_status_headers",
        ":tf_tensor_hdrs",
        "//machina/core/platform:status",
        "//machina/core/platform:tstring",
        "@local_xla//xla/tsl/c:tsl_status",
    ],
)

cc_library(
    name = "tf_file_statistics",
    hdrs = ["tf_file_statistics.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensor_interface",
    hdrs = ["tensor_interface.h"],
    visibility = ["//machina:internal"],
    deps = select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core:lib",
            "//machina/core:protos_all_cc",
        ],
    }),
)

cc_library(
    name = "tf_datatype_hdrs",
    hdrs = ["tf_datatype.h"],
    visibility = ["//machina:internal"],
    deps = [
        ":c_api_macros_hdrs",
    ],
)

cc_library(
    name = "tf_datatype",
    srcs = ["tf_datatype.cc"],
    hdrs = ["tf_datatype.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core:framework",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "tf_tensor_hdrs",
    hdrs = [
        "tf_tensor.h",
        "tf_tensor_helper.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        ":tf_datatype_hdrs",
        ":tf_status_headers",
        "//machina/core/platform:status",
    ],
)

cc_library(
    name = "tf_tensor",
    srcs = ["tf_tensor.cc"],
    hdrs = [
        "tf_tensor.h",
        "tf_tensor_helper.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros",
        ":tensor_interface",
        ":tf_datatype",
        ":tf_status",
        ":tf_status_helper",
        ":tf_tensor_internal",
        "//machina/core/platform:status",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core:framework",
            "//machina/core:lib",
            "//machina/core:protos_all_cc",
            "//machina/core/platform:casts",
        ],
    }),
)

tf_cuda_library(
    name = "tf_tensor_internal",
    hdrs = [
        "tf_tensor.h",
        "tf_tensor_helper.h",
        "tf_tensor_internal.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros",
        ":tensor_interface",
        ":tf_datatype",
        ":tf_status",
        "//machina/core/platform:status",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core:framework",
            "//machina/core:protos_all_cc",
            "//machina/core/platform:casts",
        ],
    }),
)

cc_library(
    name = "tf_buffer_hdrs",
    hdrs = [
        "tf_buffer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
    ],
)

cc_library(
    name = "tf_buffer",
    srcs = [
        "tf_buffer.cc",
    ],
    hdrs = [
        "tf_buffer.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        ":tf_buffer_internal",
        "//machina/core/platform:errors",
        "//machina/core/platform:platform_port",
        "//machina/core/platform:protobuf",
        "//machina/core/platform:status",
    ],
)

tf_cuda_library(
    name = "tf_buffer_internal",
    hdrs = [
        "tf_buffer.h",
        "tf_buffer_internal.h",
    ],
    visibility = [
        "//machina:internal",
        "//machina/c:__subpackages__",
    ],
    deps = [
        ":c_api_macros_hdrs",
        "//machina/core/platform:protobuf",
        "//machina/core/platform:status",
    ],
)

tf_cuda_library(
    name = "c_api_experimental",
    srcs = [
        "c_api_experimental.cc",
    ],
    hdrs = [
        "c_api_experimental.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api",
        ":c_api_internal",
        ":c_api_macros_hdrs",
        ":checkpoint_reader",
        ":tf_buffer",
        ":tf_buffer_internal",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_internal",
        "//machina/c/eager:tfe_context_internal",
        "//machina/c/eager:tfe_op_internal",
        "//machina/c/eager:tfe_tensorhandle_internal",
        "//machina/compiler/jit:flags",
        "//machina/compiler/jit:get_compiler_ir",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/common_runtime/eager:attr_builder",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/common_runtime/eager:core",
        "//machina/core/common_runtime/eager:eager_operation",
        "//machina/core/common_runtime/pluggable_device:pluggable_device_plugin_init",
        "//machina/core/distributed_runtime/rpc:grpc_server_lib",
        "//machina/core/platform",
        "//machina/core/platform:blocking_counter",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/c:tsl_status_internal",
    ],
    alwayslink = 1,
)

exports_files(
    [
        "version_script.lds",
        "exported_symbols.lds",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "checkpoint_reader_hdrs",
    srcs = [
        "checkpoint_reader.h",
        "tf_status_helper.h",
    ],
    visibility = ["//machina:__subpackages__"],
)

tf_cuda_library(
    name = "tf_status_helper",
    srcs = ["tf_status_helper.cc"],
    hdrs = ["tf_status_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf_status_headers",
        "@local_xla//xla/tsl/c:tsl_status_helper",
        "@local_xla//xla/tsl/platform:status",
    ],
)

tf_cc_test(
    name = "tf_status_helper_test",
    srcs = ["tf_status_helper_test.cc"],
    deps = [
        ":tf_status",
        ":tf_status_helper",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:test",
        "@local_xla//xla/tsl/platform:test_main",
    ],
)

tf_cuda_library(
    name = "checkpoint_reader",
    srcs = ["checkpoint_reader.cc"],
    hdrs = ["checkpoint_reader.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tf_status",
        ":tf_status_helper",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/util/tensor_bundle",
        "//machina/core/util/tensor_bundle:naming",
    ],
)

tf_cuda_library(
    name = "env",
    srcs = [
        "env.cc",
    ],
    hdrs = [
        "env.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:framework",
        ],
    }) + [
        ":c_api_macros",
        ":tf_file_statistics",
        ":tf_status",
        ":tf_status_helper",
        "//machina/core/platform:env",
        "//machina/core/platform:file_statistics",
        "//machina/core/platform:path",
        "//machina/core/platform:status",
        "//machina/core/platform:stringpiece",
        "//machina/core/platform:types",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "kernels_hdrs",
    hdrs = [
        "kernels.h",
        "kernels_experimental.h",
    ],
    visibility = ["//machina:internal"],
    deps = [
        ":c_api_headers",
        ":c_api_macros_hdrs",
        ":tf_buffer_hdrs",
        ":tf_datatype_hdrs",
        ":tf_status_headers",
        ":tf_tensor_hdrs",
        "//machina/c/experimental/stream_executor:stream_executor_hdrs",
        "//machina/core/common_runtime/next_pluggable_device/c:tf_rendezvous_c_api",
        "@local_tsl//tsl/platform",
    ],
)

tf_cuda_library(
    name = "kernels",
    srcs = [
        "kernels.cc",
    ],
    hdrs = [
        "kernels.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_internal",
        ":c_api_macros_hdrs",
        ":tf_buffer",
        ":tf_buffer_internal",
        ":tf_status",
        ":tf_status_helper",
        ":tf_tensor_internal",
        "//machina/core/common_runtime/next_pluggable_device/c:tf_rendezvous_c_api",
        "//machina/core/common_runtime/next_pluggable_device/c:tf_rendezvous_c_api_internal",
        "//machina/core/platform:notification",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/c:tsl_status_internal",
    ] + select({
        "//machina:android": [
            "//machina/c/experimental/stream_executor:stream_executor_hdrs",
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            ":tf_tensor",
            "//machina/c/experimental/stream_executor",
            "//machina/c/experimental/stream_executor:stream_executor_internal",
            "//machina/core:framework",
            "//machina/core:framework_lite",
            "//machina/core:protos_all_cc",
            "@local_xla//xla/stream_executor:stream",
            "@local_xla//xla/tsl/framework:device_id_utils",
            "@local_xla//xla/tsl/platform:statusor",
        ],
    }),
)

cc_library(
    name = "kernels_experimental_hdrs",
    hdrs = ["kernels_experimental.h"],
    visibility = ["//machina:internal"],
    deps = [
        ":c_api_macros_hdrs",
        ":kernels_hdrs",
    ],
)

tf_cuda_library(
    name = "kernels_experimental",
    srcs = ["kernels_experimental.cc"],
    hdrs = ["kernels_experimental.h"],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        ":kernels",
        ":tf_status_helper",
        ":tf_status_internal",
        ":tf_tensor_internal",
        "//machina/core:framework",
        "//machina/core:framework_internal_impl",
        "//machina/core/lib/gtl:cleanup",
        "//machina/core/platform:errors",
        "//machina/core/platform:mutex",
        "//machina/core/platform:refcount",
    ] + if_not_mobile([
        "//machina/core/kernels:tensor_list",
        "//machina/core/kernels:tensor_list_util",
        "//machina/core/kernels:variant_ops_util",
        "//machina/core/kernels/data:optional_ops_util",
        "//machina/core/platform:abi",
    ]),
    alwayslink = 1,
)

tf_cuda_library(
    name = "ops",
    srcs = [
        "ops.cc",
    ],
    hdrs = [
        "ops.h",
    ],
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_macros_hdrs",
        ":tf_datatype",
        ":tf_status_helper",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:framework",
        ],
    }) + if_libtpu_tf_status(),
    alwayslink = 1,
)

cc_library(
    name = "ops_hdrs",
    hdrs = ["ops.h"],
    visibility = ["//machina:internal"],
    deps = [
        ":c_api_macros_hdrs",
        ":tf_datatype",
        ":tf_status_headers",
    ],
)

# -----------------------------------------------------------------------------
# Tests

tf_cuda_library(
    name = "c_test_util",
    testonly = 1,
    srcs = ["c_test_util.cc"],
    hdrs = ["c_test_util.h"],
    visibility = [
        "//learning/brain:__subpackages__",
        "//machina:__subpackages__",
    ],
    deps = [
        ":c_api",
        ":c_api_experimental",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:session_options",
        "//machina/core:test",
    ],
)

tf_cc_test(
    name = "c_test",
    srcs = ["c_test.c"],
    extra_copts = ["-std=c11"],
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":env",
        ":kernels",
    ],
)

tf_cuda_cc_test(
    name = "c_api_test",
    size = "medium",
    srcs = ["c_api_test.cc"],
    data = [
        ":test_op1.so",
        "//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    extra_copts = if_pywrap(["-DMACHINA_NO_SHARED_OBJECTS"]),
    linkopts = select({
        "//machina:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    tags = [
        "no_cuda_asan",  # TODO(b/181771536)
        "no_windows",  # TODO(b/155444728)
    ],
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    deps = [
        ":c_api",
        ":c_api_internal",
        ":c_test_util",
        ":test_op_kernel",
        ":tf_buffer",
        ":tf_buffer_internal",
        "//machina/cc:cc_ops",
        "//machina/cc:grad_ops",
        "//machina/cc/saved_model:signature_constants",
        "//machina/cc/saved_model:tag_constants",
        "//machina/compiler/jit",
        "//machina/core:array_ops_op_lib",
        "//machina/core:bitwise_ops_op_lib",
        "//machina/core:control_flow_ops_op_lib",
        "//machina/core:core_cpu_internal",
        "//machina/core:direct_session",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:functional_ops_op_lib",
        "//machina/core:lib",
        "//machina/core:math_ops_op_lib",
        "//machina/core:nn_ops_op_lib",
        "//machina/core:no_op_op_lib",
        "//machina/core:protos_all_cc",
        "//machina/core:sendrecv_ops_op_lib",
        "//machina/core:spectral_ops_op_lib",
        "//machina/core:state_ops_op_lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/kernels:array",
        "//machina/core/kernels:control_flow_ops",
        "//machina/core/kernels:math",
        "//machina/core/platform:resource_loader",
    ],
)

tf_cc_test(
    name = "c_api_experimental_test",
    size = "medium",
    srcs = ["c_api_experimental_test.cc"],
    data = [
        "testdata/tf_record",
        "//machina/c/experimental/stream_executor/test:test_pluggable_device.so",
        "//machina/core/common_runtime/next_pluggable_device/c:test_next_pluggable_device_plugin.so",
    ],
    extra_copts = if_google(["-DMACHINA_NO_SHARED_OBJECTS=1"]),
    linkopts = select({
        "//machina:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    deps = [
        ":c_api",
        ":c_api_experimental",
        ":c_api_internal",
        ":c_test_util",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_test_util",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:resource_loader",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "c_api_function_test",
    size = "medium",
    srcs = ["c_api_function_test.cc"],
    deps = [
        ":c_api",
        ":c_api_internal",
        ":c_test_util",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

tf_cc_test(
    name = "while_loop_test",
    size = "medium",
    srcs = ["while_loop_test.cc"],
    deps = [
        ":c_api",
        ":c_test_util",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

tf_custom_op_library(
    name = "test_op1.so",
    srcs = ["test_op1.cc"],
)

tf_kernel_library(
    name = "test_op_kernel",
    srcs = ["test_op.cc"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
    ],
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "env_test",
    size = "medium",
    srcs = ["env_test.cc"],
    linkopts = select({
        "//machina:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    deps = [
        ":c_api",
        ":env",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

tf_cuda_cc_test(
    name = "kernels_test",
    size = "medium",
    srcs = ["kernels_test.cc"],
    linkopts = select({
        "//machina:macos": ["-headerpad_max_install_names"],
        "//conditions:default": [],
    }),
    tags = ["no_cuda_on_cpu_tap"],
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    deps = [
        ":c_api",
        ":kernels",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/kernels:ops_testutil",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings:str_format",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "ops_test",
    size = "medium",
    srcs = ["ops_test.cc"],
    linkopts = select({
        "//conditions:default": [],
    }),
    # We must ensure that the dependencies can be dynamically linked since
    # the shared library must be able to use core:framework.
    deps = [
        ":c_api",
        ":ops",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

# -----------------------------------------------------------------------------
# Python API target

tf_cuda_library(
    name = "python_api",
    srcs = ["python_api.cc"],
    hdrs = ["python_api.h"],
    visibility = [
        "//machina:internal",
        "//machina/python:__pkg__",
    ],
    deps = [
        ":c_api",
        ":c_api_internal",
        "//machina/core:protos_all_cc",
        # TODO(b/74620627): remove when _USE_C_SHAPES is removed
    ],
    alwayslink = 1,
)

cc_library(
    name = "conversion_macros",
    hdrs = [
        "conversion_macros.h",
    ],
    visibility = ["//machina:__subpackages__"],
)

cc_library(
    name = "c_op_requires",
    hdrs = ["c_op_requires.h"],
    visibility = ["//visibility:public"],
    deps = ["//machina/core/platform:macros"],
)
