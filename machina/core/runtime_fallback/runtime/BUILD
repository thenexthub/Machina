load("//machina/core/platform:rules_cc.bzl", "cc_library")

package_group(
    name = "internal",
    packages = [
        "//machina/core/runtime_fallback/runtime/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

cc_library(
    name = "op_logger",
    hdrs = [
        "op_logger.h",
    ],
    visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt/cpp_tests/saved_model:__subpackages__",
        "//machina/core/runtime_fallback:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/memory",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

cc_library(
    name = "runtime_fallback_alwayslink",
    srcs = [
        "conversion_function.cc",
        "runtime_fallback_batch_tf_opkernels.cc",
        "runtime_fallback_kernels.cc",
        "runtime_fallback_op_handler.cc",
        "static_registration.cc",
    ],
    hdrs = [
        "conversion_function.h",
        "runtime_fallback_kernels.h",
        "runtime_fallback_op_handler.h",
        "runtime_fallback_tensor.h",
    ],
    includes = [
        "third_party/tf_runtime/include",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":fallback_batch_kernel",
        ":kernel_utils",
        ":op_logger",
        ":runtime_fallback_tensor",
        "//machina/core/kernels/batching_util:adaptive_shared_batch_scheduler",
        "//machina/core/kernels/batching_util:batch_scheduler_hdrs",
        "//machina/core/platform:statusor",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_execute_compat",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "//machina/core/runtime_fallback/util:attr_util",
        "//machina/core/runtime_fallback/util:tensor_util",
        "//machina/core/runtime_fallback/util:type_util",
        "//machina/core/tfrt/runtime:work_queue_interface",
        "//machina/core/tfrt/utils:error_util",
        "//machina/core/tfrt/utils:fallback_tensor",
        "//machina/core/tfrt/utils:tensor_util",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/cpu:core_runtime",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/c:c_api_internal",
            "//machina/c:tensor_interface",
            "//machina/c:tf_datatype",
            "//machina/c:tf_status",
            "//machina/c:tf_tensor_internal",
            "//machina/c/eager:abstract_operation",
            "//machina/c/eager:abstract_tensor_handle",
            "//machina/core:framework",
            "//machina/core:protos_all_cc",
            "//machina/core:session_options",
            "//machina/core/common_runtime:core_cpu_impl",
            "//machina/core/common_runtime/eager:context",
            "//machina/core/common_runtime/eager:core",  # Needed due to circular dep
            "//machina/core/common_runtime/eager:eager_operation",
            "//machina/core/common_runtime/eager:execute",
            "//machina/core/common_runtime/eager:tensor_handle",
            "//machina/core/framework:node_def_proto_cc",
            "//machina/core/framework:tensor",
            "//machina/core/kernels/batching_util:batch_resource_base",
            "//machina/core/kernels/batching_util:bounded_executor",
            "//machina/core/platform:errors",
            "//machina/core/platform:random",
            "//machina/core/platform:status",
            "//machina/core/profiler/lib:traceme",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "runtime_fallback_tensor",
    srcs = ["runtime_fallback_tensor.cc"],
    hdrs = ["runtime_fallback_tensor.h"],
    visibility = [
        ":internal",
        "//machina/core/runtime_fallback/conversion:__subpackages__",
    ],
    deps = [
        ":kernel_utils",
        "//machina/core/runtime_fallback/util:tensor_util",
        "//machina/core/runtime_fallback/util:type_util",
        "@com_google_absl//absl/status",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/c:tensor_interface",
            "//machina/c:tf_datatype",
            "//machina/c:tf_tensor",
            "//machina/c:tf_tensor_internal",
            "//machina/core:protos_all_cc",
            "//machina/core/common_runtime/eager:tensor_handle",
            "//machina/core/framework:tensor",
        ],
    }),
)

cc_library(
    name = "kernel_utils",
    srcs = ["kernel_utils.cc"],
    hdrs = ["kernel_utils.h"],
    visibility = [
        ":internal",
        "//machina/compiler/mlir/tfrt:__pkg__",
        "//machina/core/runtime_fallback:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/c:tf_tensor",
            "//machina/core:protos_all_cc",
            "//machina/core:session_options",
            "//machina/core/common_runtime:device_mgr",
            "//machina/core/common_runtime/eager:context",
            "//machina/core/common_runtime/eager:context_distributed_manager",
            "//machina/core/common_runtime/eager:core_no_xla",
            "//machina/core/common_runtime/eager:eager_operation",
            "//machina/core/common_runtime/eager:tensor_handle",
            "//machina/core/platform:status",
        ],
    }),
)

cc_library(
    name = "fallback_batch_kernel",
    srcs = ["fallback_batch_kernel.cc"],
    hdrs = ["fallback_batch_kernel.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/framework:op_requires",
        "//machina/core/kernels:batch_kernels",
        "//machina/core/kernels/batching_util:adaptive_shared_batch_scheduler",
        "//machina/core/kernels/batching_util:batch_resource_base",
        "//machina/core/kernels/batching_util:batch_scheduler_hdrs",
        "//machina/core/kernels/batching_util:batch_stats",
        "//machina/core/kernels/batching_util:bounded_executor",
        "//machina/core/kernels/batching_util:warmup",
        "//machina/core/lib/core:refcount",
        "//machina/core/lib/core:status",
        "//machina/core/platform:errors",
        "//machina/core/platform:random",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//machina/core/tfrt/fallback:op_kernel_runner",
        "//machina/core/tfrt/utils:error_util",
        "//machina/core/tfrt/utils:fallback_tensor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@tf_runtime//:hostcontext",
    ],
)
