load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        # copybara:uncomment "//smartass/brain/ops/tfrt_kernels:__subpackages__",
        "//machina/compiler/mlir/tfrt/transforms/mlrt:__subpackages__",
        "//machina/core/tfrt:__subpackages__",
    ],
)

cc_library(
    name = "mlir_to_bytecode",
    srcs = ["mlir_to_bytecode.cc"],
    hdrs = ["mlir_to_bytecode.h"],
    deps = [
        "//machina/core/tfrt/mlrt/bytecode",
        "//machina/core/tfrt/mlrt/bytecode:executable",
        "//machina/core/tfrt/mlrt/bytecode:function",
        "//machina/core/tfrt/mlrt/bytecode:kernel",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "mlir_to_bytecode_test",
    srcs = ["mlir_to_bytecode_test.cc"],
    data = glob(["testdata/**"]),
    deps = [
        ":mlir_to_bytecode",
        "//machina/core/tfrt/mlrt/bytecode",
        "//machina/core/tfrt/mlrt/bytecode:executable",
        "//machina/core/tfrt/mlrt/interpreter:attribute_span",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Parser",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/tsl/platform:resource_loader",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/platform:test",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        # copybara:uncomment "//learning/brain/experimental/tfrt/native_lowering/stubs:tfrt_native_lowering_impl",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
        "//machina/core:core_cpu_base",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core/tfrt/graph_executor:sync_resource_state",
        "//machina/core/tfrt/mlrt/attribute",
        "//machina/core/tfrt/mlrt/bytecode",
        "//machina/core/tfrt/mlrt/bytecode:kernel",
        "//machina/core/tfrt/mlrt/interpreter:context",
        "//machina/core/tfrt/mlrt/interpreter:interpreter_testutil",
        "//machina/core/tfrt/mlrt/interpreter:value",
        "//machina/core/tfrt/stubs:tfrt_native_lowering_stub",
        "//machina/core/tfrt/utils:tensor_util",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
    ],
)
