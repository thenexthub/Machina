load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

cc_library(
    name = "rewrites",
    srcs = [
        "func.cc",
        "general.cc",
        "scf.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/mlir/tools/mlir_bisect:bisect_lib",
        "//xla/mlir/tools/mlir_replay/public:execution_trace_utils",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:DialectUtils",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:SideEffectInterfaces",
        "@toolchain-project//mlir:Support",
    ],
    alwayslink = 1,
)
