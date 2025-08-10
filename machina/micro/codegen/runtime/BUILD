load("//machina/lite/micro:build_def.bzl", "tflm_cc_library")

package(default_visibility = ["//visibility:public"])

tflm_cc_library(
    name = "micro_codegen_context",
    srcs = ["micro_codegen_context.cc"],
    hdrs = ["micro_codegen_context.h"],
    deps = [
        "//machina/lite/c:common",
        "//machina/lite/kernels:op_macros",
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/micro:micro_context",
        "//machina/lite/micro:micro_graph",
        "//machina/lite/micro:micro_log",
        "//machina/lite/micro:span",
    ],
)
