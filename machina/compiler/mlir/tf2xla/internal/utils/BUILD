load("//machina:machina.bzl", "tf_cc_test")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/compiler/mlir/tf2xla/internal:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "dialect_detection_utils",
    srcs = [
        "dialect_detection_utils.cc",
    ],
    hdrs = [
        "dialect_detection_utils.h",
    ],
    deps = [
        "//machina/compiler/mlir/machina",
        "//machina/core:framework",
        "//machina/core/transforms/toposort:Pass",
        "@com_google_absl//absl/strings",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:Transforms",
    ],
)

tf_cc_test(
    name = "dialect_detection_utils_test",
    srcs = ["dialect_detection_utils_test.cc"],
    deps = [
        ":dialect_detection_utils",
        "//machina/compiler/mlir/machina:machina_types",
        "@com_google_googletest//:gtest_main",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Pass",
        "@stablehlo//:chlo_ops",
    ],
)

cc_library(
    name = "test_metadata_config",
    testonly = True,
    srcs = ["test_metadata_config.cc"],
    hdrs = ["test_metadata_config.h"],
    visibility = [
        "//machina/compiler/mlir/tf2xla/api:__subpackages__",
    ],
    deps = [
        "//machina/compiler/jit",
        "//machina/compiler/jit:xla_tpu_device",
        "//machina/compiler/mlir:register_common_dialects",
        "//machina/compiler/mlir/machina:convert_type",
        "//machina/compiler/mlir/machina:deserialize_mlir_module_utils",
        "//machina/compiler/tf2xla:layout_util",
        "//machina/compiler/tf2xla:xla_compiler",
        "//machina/compiler/tf2xla:xla_helpers",
        "//machina/compiler/tf2xla:xla_tpu_backend_registration",
        "//machina/compiler/tf2xla/kernels:xla_ops",
        "//machina/core:framework",
        "//machina/core:test_main",
        "//machina/core/protobuf/tpu:compile_metadata_proto_cc",
        "//machina/core/tpu/kernels/xla:host_compute_ops",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "@local_xla//xla/mlir_hlo:hlo_dialect_registration",
        "@local_xla//xla/tsl/platform:errors",
    ],
)
