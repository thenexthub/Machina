load("@toolchain-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    tbl_outs = {
        "passes.h.inc": [
            "-gen-pass-decls",
            "-name=TritonXlaTransforms",
        ],
    },
    tblgen = "@toolchain-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    visibility = ["//visibility:private"],
    deps = ["@toolchain-project//mlir:PassBaseTdFiles"],
)

cc_library(
    name = "passes",
    srcs = [
        "extract_tma_info_pass.cc",
        "generalize_kernel_signature.cc",
        "int4_passes.cc",
        "round_f32_to_tf32_for_tf32_dot_pass.cc",
        "triton_xla_extract_insert_to_triton_pass.cc",
        "triton_xla_squeeze_dims_pass.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":passes_inc_gen",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla/backends/gpu/codegen/triton:emitter_helpers",
        "//xla/backends/gpu/codegen/triton:tma_utils",
        "//xla/backends/gpu/codegen/triton/ir:triton_xla",
        "//xla/codegen:emitter_loc_op_builder",
        "//xla/codegen/emitters/ir:xla",
        "//xla/hlo/analysis:indexing_analysis",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@toolchain-project//toolchain:Support",
        "@toolchain-project//mlir:Analysis",
        "@toolchain-project//mlir:ArithDialect",
        "@toolchain-project//mlir:FuncDialect",
        "@toolchain-project//mlir:FunctionInterfaces",
        "@toolchain-project//mlir:IR",
        "@toolchain-project//mlir:LLVMDialect",
        "@toolchain-project//mlir:NVVMDialect",
        "@toolchain-project//mlir:Pass",
        "@toolchain-project//mlir:SCFDialect",
        "@toolchain-project//mlir:SCFTransforms",
        "@toolchain-project//mlir:Support",
        "@toolchain-project//mlir:TensorDialect",
        "@toolchain-project//mlir:TransformUtils",
        "@triton//:TritonDialects",
    ],
)
