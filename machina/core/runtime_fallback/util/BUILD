load(
    "//machina:machina.bzl",
    "if_oss",
    "tf_cc_test",
    "tf_cuda_library",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package_group(
    name = "internal",
    packages = [
        "//learning/brain/experimental/tfrt/native_lowering/kernels/...",
        "//machina/compiler/mlir/tfrt/tests/tfrt_fallback/...",
        "//machina/core/runtime_fallback/...",
        "//machina/core/tfrt/utils/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

cc_library(
    name = "attr_util",
    srcs = [
        "attr_util.cc",
    ],
    hdrs = [
        "attr_type.def",
        "attr_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "//machina/core/tfrt/utils:tensor_util",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:bef",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core:framework",
            "//machina/core:protos_all_cc",
            "//machina/core/framework:tensor",
            "//machina/core/framework:tensor_proto_cc",
            "//machina/core/framework:types_proto_cc",
        ],
    }),
)

cc_library(
    name = "type_util",
    hdrs = [
        "dtype.def",
        "type_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core/platform:logging",
        "@toolchain-project//toolchain:Support",
        "@tf_runtime//:dtype",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/core/framework:types_proto_cc",
        ],
    }),
)

cc_library(
    name = "fallback_test_util",
    srcs = ["fallback_test_util.cc"],
    hdrs = ["fallback_test_util.h"],
    tags = ["no_oss"],
    deps = [
        "//machina/c:tf_tensor",
        "//machina/core:framework",
        "//machina/core/platform:threadpool_interface",
        "//machina/core/runtime_fallback/kernel:kernel_fallback_execute_compat",
        "//machina/core/runtime_fallback/runtime:kernel_utils",
        "@tf_runtime//:hostcontext",
    ],
)

tf_cuda_library(
    name = "tensor_metadata",
    hdrs = ["tensor_metadata.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":type_util",
        "//machina/core/framework:tensor",
        "//machina/core/framework:tensor_shape",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)

tf_cuda_library(
    name = "tensor_util",
    srcs = ["tensor_util.cc"],
    hdrs = [
        "tensor_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":tensor_metadata",
        ":type_util",
        "//machina/c:tf_datatype",
        "//machina/core/framework:tensor_shape",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/platform:tstring",
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
            "//machina/c:tf_tensor_internal",
            "//machina/core/framework:tensor",
            "//machina/core/platform:status",
        ],
    }),
)

tf_cc_test(
    name = "type_util_test",
    srcs = ["type_util_test.cc"],
    deps = [
        ":type_util",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
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

tf_cc_test(
    name = "tensor_util_test",
    srcs = ["tensor_util_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":tensor_util",
        "//machina/c:tf_tensor",
        "//machina/core/framework:tensor",
        "//machina/core/framework:tensor_proto_cc",
        "//machina/core/framework:tensor_shape",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/platform:protobuf",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
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
            "//machina/core:test",
            "//machina/core:test_main",
            "//machina/core/platform:status",
            "//machina/core/platform:types",
        ],
    }),
)

tf_cc_test(
    name = "attr_util_test",
    srcs = ["attr_util_test.cc"],
    tags = if_oss([
        "manual",
        "no_oss",
    ]),  # b/169705709, no protobuf matchers in OSS.
    deps = [
        ":attr_util",
        "//machina/c:tf_tensor",
        "//machina/core/framework:attr_value_proto_cc",
        "//machina/core/framework:node_def_util",
        "//machina/core/framework:tensor",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/platform:errors",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:bef",
        "@tf_runtime//:bef_attr_encoder",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:dtype",
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
