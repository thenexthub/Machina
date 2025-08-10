# Description:
#    TF2XLA Bridge and related components.

load("//machina/core/platform:rules_cc.bzl", "cc_library")

package_group(
    name = "machina_mlir_tf2xla",
    packages = [
        "//machina/compiler/mlir/tf2xla/...",
    ],
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "mlir_bridge_rollout_policy",
    srcs = ["mlir_bridge_rollout_policy.cc"],
    hdrs = ["mlir_bridge_rollout_policy.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/compiler/jit:flags",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/types:optional",
        "@toolchain-project//mlir:IR",
    ],
)

alias(
    name = "compile_mlir_util",
    actual = "//machina/compiler/mlir/tf2xla/api/v1:compile_mlir_util_no_tf_dialect_passes",
)
