load("//machina:machina.bzl", "tf_cc_test")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/compiler/mlir/tf2xla/api/v2:__subpackages__",
    ],
)

cc_library(
    name = "compile_mlir",
    testonly = True,
    srcs = ["compile_mlir.cc"],
    hdrs = ["compile_mlir.h"],
    deps = [
        "//machina/compiler/jit",
        "//machina/compiler/mlir/tf2xla/api/v2:legalize_tf",
        "//machina/compiler/mlir/tf2xla/internal/utils:test_metadata_config",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_helpers",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/core/protobuf/tpu:compile_metadata_proto_cc",
        "//machina/core/tpu/kernels:tpu_compile_op_support",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//mlir:Pass",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/stream_executor:platform",
        "@local_xla//xla/stream_executor:platform_manager",
    ],
)

cc_library(
    name = "utils",
    testonly = True,
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = ["//machina/core/platform:resource_loader"],
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "@com_google_googletest//:gtest_main",
    ],
)
