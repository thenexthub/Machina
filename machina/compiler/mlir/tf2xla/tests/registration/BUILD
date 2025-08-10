load("//machina:machina.bzl", "tf_cc_test")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "graph_to_tf_executor_registration",
    srcs = [
        "graph_to_tf_executor_registration.cc",
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TranslateLib",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:compile_only_client",
        "@local_xla//xla/service/cpu:cpu_compiler",
        "@local_xla//xla/service/cpu:cpu_transfer_manager",
        "@local_xla//xla/stream_executor:platform_manager",
        "@local_xla//xla/stream_executor/host:host_platform",
        "@local_xla//xla/stream_executor/host:host_platform_id",
        "@local_xla//xla/tsl/platform:status",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "graph_to_tf_executor_registration_test",
    size = "small",
    srcs = ["graph_to_tf_executor_registration_test.cc"],
    deps = [
        ":graph_to_tf_executor_registration",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TranslateLib",
    ],
)
