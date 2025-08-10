load("//machina:machina.default.bzl", "tf_gen_op_wrapper_cc")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/compiler/tf2xla:friends"],
    licenses = ["notice"],
)

tf_gen_op_wrapper_cc(
    name = "xla_ops_gen",
    out_ops_file = "ops/xla_ops",
    deps = ["//machina/compiler/tf2xla/ops:xla_ops"],
)

cc_library(
    name = "xla_ops",
    srcs = ["ops/xla_ops.cc"],
    hdrs = ["ops/xla_ops.h"],
    deps = [
        "//machina/cc:const_op",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/compiler/tf2xla/ops:xla_ops",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ],
)

tf_gen_op_wrapper_cc(
    name = "xla_jit_op_gen",
    include_internal_ops = 1,
    out_ops_file = "ops/xla_jit_ops",
    deps = ["//machina/compiler/jit/ops:xla_ops"],
)

cc_library(
    name = "xla_jit_ops",
    srcs = ["ops/xla_jit_ops.cc"],
    hdrs = ["ops/xla_jit_ops.h"],
    deps = [
        "//machina/cc:const_op",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/compiler/jit/ops:xla_ops",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ],
)
