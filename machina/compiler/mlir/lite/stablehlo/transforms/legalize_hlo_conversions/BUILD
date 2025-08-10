# Groups the implementations of the HLO to TF operation conversions.
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "util",
    srcs = [
        "util.cc",
    ],
    hdrs = [
        "util.h",
    ],
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "scatter",
    srcs = [
        "scatter.cc",
    ],
    hdrs = [
        "scatter.h",
    ],
    deps = [
        ":util",
        "//machina/compiler/mlir/machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "reduce",
    srcs = [
        "reduce.cc",
    ],
    hdrs = [
        "reduce.h",
    ],
    deps = [
        ":util",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/stablehlo:hlo_matchers",
        "//machina/compiler/mlir/machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "dot_general",
    srcs = [
        "dot_general.cc",
    ],
    hdrs = ["dot_general.h"],
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "custom_call",
    srcs = [
        "custom_call.cc",
    ],
    hdrs = [
        "custom_call.h",
    ],
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//mlir:AsmParser",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "gelu",
    srcs = ["gelu.cc"],
    hdrs = ["gelu.h"],
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "conv",
    srcs = ["conv.cc"],
    hdrs = ["conv.h"],
    deps = [
        ":conv_util",
        ":op_util_common",
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "conv_util",
    srcs = ["conv_util.cc"],
    hdrs = ["conv_util.h"],
    deps = [
        ":op_util_common",
        "//machina/core/lib/math:math_util",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "pad",
    srcs = ["pad.cc"],
    hdrs = ["pad.h"],
    deps = [
        ":op_util_common",
        ":pad_util",
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "pad_util",
    srcs = ["pad_util.cc"],
    hdrs = ["pad_util.h"],
    deps = [
        ":op_util_common",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "gather",
    srcs = ["gather.cc"],
    hdrs = ["gather.h"],
    deps = [
        ":util",
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "reduce_window",
    srcs = ["reduce_window.cc"],
    hdrs = ["reduce_window.h"],
    deps = [
        ":op_util_common",
        ":reduce_window_util",
        ":util",
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "op_util_common",
    srcs = ["op_util_common.cc"],
    hdrs = ["op_util_common.h"],
    deps = [
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
    ],
)

cc_library(
    name = "reduce_window_util",
    srcs = ["reduce_window_util.cc"],
    hdrs = ["reduce_window_util.h"],
    deps = [
        ":op_util_common",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "slice",
    srcs = ["slice.cc"],
    hdrs = ["slice.h"],
    deps = [
        ":op_util_common",
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "sort",
    srcs = ["sort.cc"],
    hdrs = ["sort.h"],
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/lite/stablehlo:hlo_matchers",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "iota",
    srcs = ["iota.cc"],
    hdrs = ["iota.h"],
    deps = [
        ":op_util_common",
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "while",
    srcs = ["while.cc"],
    hdrs = ["while.h"],
    deps = [
        ":op_util_common",
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/machina",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "get_dimension_size",
    srcs = ["get_dimension_size.cc"],
    hdrs = ["get_dimension_size.h"],
    deps = [
        ":util",
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "if",
    srcs = ["if.cc"],
    hdrs = ["if.h"],
    deps = [
        ":util",
        "//machina/compiler/mlir/lite:machina_lite",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)

cc_library(
    name = "fft",
    srcs = ["fft.cc"],
    hdrs = ["fft.h"],
    deps = [
        "//machina/compiler/mlir/lite:machina_lite",
        "//machina/compiler/mlir/machina:dynamic_shape_utils",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TransformUtils",
        "@local_xla//xla/mlir_hlo",
    ],
)
