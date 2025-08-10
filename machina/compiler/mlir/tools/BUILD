load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "translate_cl_options",
    srcs = [
        "tf_mlir_translate_cl.cc",
    ],
    hdrs = [
        "tf_mlir_translate_cl.h",
    ],
    deps = [
        "@toolchain-project//toolchain:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "translate_registration",
    srcs = [
        "tf_mlir_translate_registration.cc",
    ],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/compiler/mlir/machina/translate:mlir_roundtrip_flags",
        "//machina/compiler/mlir/machina/translate/tools:file_tf_mlir_translate",
        "//machina/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//machina/compiler/mlir/tools:translate_cl_options",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla/kernels:xla_ops",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:TranslateLib",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:compile_only_client",
        "@local_xla//xla/service/cpu:cpu_compiler",
        "@local_xla//xla/service/cpu:cpu_transfer_manager",
        "@local_xla//xla/stream_executor/host:host_platform",
        "@local_xla//xla/stream_executor/host:host_platform_id",
    ],
    alwayslink = 1,
)

cc_library(
    name = "safe_cast",
    hdrs = ["safe_cast.h"],
)
