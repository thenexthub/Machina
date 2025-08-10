# Fallback from TFRT to TF kernels and related utilities.
load(
    "//machina:machina.bzl",
    "tf_cc_shared_test",
    "tf_cc_test",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/core/runtime_fallback:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "tfrt_op_kernel",
    srcs = ["tfrt_op_kernel.cc"],
    hdrs = ["tfrt_op_kernel.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":attr_util",
        "//machina/core/runtime_fallback/util:attr_util",
        "//machina/core/tfrt/utils:error_util",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//backends/common:eigencompat",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core/framework:tensor",
            "//machina/core/framework:tensor_shape",
            "//machina/core/platform:errors",
            "//machina/core/platform:status",
            "//machina/core/platform:stringpiece",
        ],
    }),
)

tf_cc_shared_test(
    name = "tfrt_op_kernel_test",
    srcs = ["tfrt_op_kernel_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":tfrt_op_kernel",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/c:c_api",
            "//machina/core:test",
            "//machina/core:test_main",
            "//machina/core/framework:types_proto_cc",
            "//machina/core/lib/core:error_codes_proto_cc",
            "//machina/core/util:padding",
        ],
    }),
)

cc_library(
    name = "attr_util",
    srcs = ["attr_util.cc"],
    hdrs = ["attr_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:portable_gif_internal",
        "//machina/core/runtime_fallback/util:attr_util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core/framework:types_proto_cc",
            "//machina/core/platform:errors",
            "//machina/core/platform:status",
            "//machina/core/platform:str_util",
            "//machina/core/platform:stringpiece",
            "//machina/core/util:padding",
        ],
    }),
)

cc_library(
    name = "tensor_util",
    srcs = ["tensor_util.cc"],
    hdrs = ["tensor_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":kernel_fallback_tensor",
        "//machina/core:framework",
        "//machina/core/common_runtime:copy_tensor",
        "//machina/core/runtime_fallback/runtime:kernel_utils",
        "//machina/core/runtime_fallback/util:attr_util",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core/framework:types_proto_cc",
            "//machina/core/platform:errors",
            "//machina/core/platform:status",
            "//machina/core/platform:str_util",
            "//machina/core/platform:stringpiece",
            "//machina/core/util:padding",
        ],
    }),
)

tf_cc_test(
    name = "attr_util_test",
    srcs = ["attr_util_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":attr_util",
        "//machina/c:tf_tensor",
        "@com_google_absl//absl/status",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core:test",
            "//machina/core:test_main",
            "//machina/core/platform:status",
            "//machina/core/platform:types",
        ],
    }),
)

cc_library(
    name = "kernel_fallback_kernels_alwayslink",
    srcs = [
        "kernel_fallback_kernels.cc",
        "kernel_fallback_static_registration.cc",
    ],
    includes = [
        "third_party/tf_runtime/include",
    ],
    deps = [
        ":attr_util",
        ":kernel_fallback_execute",
        ":kernel_fallback_op_handler",
        ":kernel_fallback_tensor",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/cpu:core_runtime",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core/framework:tensor",
        ],
    }),
    alwayslink = 1,
)

cc_library(
    name = "kernel_fallback_tensor",
    srcs = ["kernel_fallback_tensor.cc"],
    hdrs = ["kernel_fallback_tensor.h"],
    visibility = [
        "//machina/core/runtime_fallback:internal",
        "//machina/core/tfrt:__subpackages__",
    ],
    deps = [
        "//machina/core/runtime_fallback/util:tensor_util",
        "//machina/core/runtime_fallback/util:type_util",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor_alwayslink",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/c:tf_datatype",
            "//machina/c:tf_tensor_internal",
            "//machina/core/framework:tensor",
            "//machina/core/framework:tensor_shape",
            "//machina/core/framework:types_proto_cc",
            "//machina/core/platform:status",
        ],
    }),
)

cc_library(
    name = "conversion_lib",
    srcs = [
        "conversion/conversion.cc",
    ],
    hdrs = [
        "conversion/conversion.h",
    ],
    includes = [
        "third_party/tf_runtime/include",
    ],
    deps = [
        ":kernel_fallback_tensor",
        ":tensor_util",
        "//machina/core:portable_gif_internal",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/runtime_fallback/util:tensor_util",
        "//machina/core/runtime_fallback/util:type_util",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor_alwayslink",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core/framework:tensor",
        ],
    }),
)

cc_library(
    name = "kernel_fallback_tensor_conversion_alwayslink",
    srcs = [
        "conversion/static_registration.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":conversion_lib",
        "@tf_runtime//:tensor",
    ],
    alwayslink = 1,
)

cc_library(
    name = "kernel_fallback_op_handler",
    srcs = ["kernel_fallback_op_handler.cc"],
    hdrs = ["kernel_fallback_op_handler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":kernel_fallback_execute_compat",
        ":kernel_fallback_tensor",
        ":kernel_fallback_tensor_conversion_alwayslink",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//machina/core/runtime_fallback/runtime:kernel_utils",
        "//machina/core/runtime_fallback/util:attr_util",
        "//machina/core/tfrt/fallback:op_kernel_runner",
        "//machina/core/tfrt/fallback:op_kernel_runner_cache",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor_alwayslink",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core/common_runtime/eager:context",
            "//machina/core/framework:tensor",
            "//machina/core/platform:mutex",
        ],
    }),
    alwayslink = True,
)

cc_library(
    name = "kernel_fallback_execute",
    srcs = ["kernel_fallback_execute.cc"],
    hdrs = ["kernel_fallback_execute.h"],
    deps = [
        ":kernel_fallback_tensor",
        ":kernel_fallback_tensor_conversion_alwayslink",
        ":tfrt_op_kernel",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor_alwayslink",
        "@tf_runtime//backends/common:eigencompat",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core/framework:tensor",
        ],
    }),
)

cc_library(
    name = "kernel_fallback_execute_compat",
    srcs = ["kernel_fallback_execute_compat.cc"],
    hdrs = ["kernel_fallback_execute_compat.h"],
    visibility = [
        "//machina/core/runtime_fallback:internal",
        "//machina/core/tfrt/graph_executor:__pkg__",
        "//machina/core/tfrt/saved_model:__pkg__",
    ],
    deps = [
        ":kernel_fallback_compat_request_state",
        ":kernel_fallback_tensor",
        ":kernel_fallback_utils",
        "//machina/core/framework:tensor_proto_cc",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/runtime_fallback/runtime:op_logger",
        "//machina/core/runtime_fallback/util:attr_util",
        "//machina/core/runtime_fallback/util:type_util",
        "//machina/core/tfrt/fallback:cost_recorder",
        "//machina/core/tfrt/fallback:device_with_custom_allocator",
        "//machina/core/tfrt/fallback:op_kernel_runner",
        "//machina/core/tfrt/fallback:op_kernel_runner_cache",
        "//machina/core/tfrt/utils",
        "//machina/core/tfrt/utils:error_util",
        "//machina/core/tfrt/utils:fallback_tensor",
        "//machina/core/tfrt/utils:tensor_util",
        "@com_google_absl//absl/base",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor_alwayslink",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core:framework",
            "//machina/core:lib",
            "//machina/core/framework:tensor",
        ],
    }),
    alwayslink = True,
)

cc_library(
    name = "kernel_fallback_execute_compat_eager",
    srcs = ["kernel_fallback_execute_compat_eager.cc"],
    hdrs = ["kernel_fallback_execute_compat_eager.h"],
    visibility = [
        "//machina/compiler/mlir/tfrt:__pkg__",
        "//machina/core/runtime_fallback:internal",
    ],
    deps = [
        ":kernel_fallback_compat_request_state",
        ":kernel_fallback_tensor",
        ":kernel_fallback_tensor_conversion_alwayslink",
        ":kernel_fallback_utils",
        "//machina/core/profiler/lib:traceme",
        "//machina/core/runtime_fallback/runtime:kernel_utils",
        "//machina/core/runtime_fallback/runtime:op_logger",
        "//machina/core/runtime_fallback/util:attr_util",
        "//machina/core/tfrt/fallback:cost_recorder",
        "//machina/core/tfrt/fallback:device_with_custom_allocator",
        "//machina/core/tfrt/fallback:op_kernel_runner",
        "//machina/core/tfrt/fallback:op_kernel_runner_cache",
        "//machina/core/tfrt/utils:error_util",
        "//machina/core/tfrt/utils:fallback_tensor",
        "//machina/core/tfrt/utils:tensor_util",
        "@com_google_absl//absl/base",
        "@toolchain-project//toolchain:Support",
        "@local_tsl//tsl/platform:refcount",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor_alwayslink",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core:framework",
            "//machina/core:lib",
            "//machina/core/common_runtime/eager:context",
            "//machina/core/common_runtime/eager:core",
            "//machina/core/framework:tensor",
        ],
    }),
    alwayslink = True,
)

cc_library(
    name = "kernel_fallback_compat_request_state",
    srcs = ["kernel_fallback_compat_request_state.cc"],
    hdrs = ["kernel_fallback_compat_request_state.h"],
    visibility = [
        "//machina/core/runtime_fallback:internal",
        # Sync fallback kernels need access to the fallback state.
        "//learning/brain/experimental/tfrt/native_lowering/kernels:__subpackages__",
        "//machina/core/tfrt/graph_executor:__subpackages__",
        "//machina/core/tfrt/saved_model:__subpackages__",
        "//machina/core/tfrt/mlrt/kernel:__subpackages__",
        "//machina/core/tfrt/gpu/kernel:__subpackages__",
    ],
    deps = [
        "//machina/core/tfrt/fallback:cost_recorder",
        "//machina/core/tfrt/fallback:op_kernel_runner",
        "//machina/core/tfrt/graph_executor:config",
        "//machina/core/tfrt/graph_executor:config_proto_cc",
        "//machina/core/tfrt/utils:fallback_tensor",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor_alwayslink",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core:core_cpu_base",
            "//machina/core:framework",
            "//machina/core:framework_lite",
            "//machina/core:lib",
            "//machina/core/common_runtime:renamed_device",
            "//machina/core/common_runtime:rendezvous_mgr",
            "//machina/core/framework:tensor",
            "//machina/core/platform:refcount",
        ],
    }),
)

cc_library(
    name = "kernel_fallback_utils",
    srcs = ["kernel_fallback_utils.cc"],
    hdrs = ["kernel_fallback_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":kernel_fallback_compat_request_state",
        "//machina/core:framework",
        "//machina/core/tfrt/fallback:op_kernel_runner",
        "//machina/core/tfrt/utils:fallback_tensor",
        "@com_google_absl//absl/container:inlined_vector",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)
