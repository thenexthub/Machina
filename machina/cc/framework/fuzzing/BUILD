# TODO(unda): describe this package.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_copts")
load(
    "//machina/cc/framework/fuzzing:op_fuzzing.bzl",
    "tf_gen_op_wrappers_fuzz",
)

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

cc_library(
    name = "cc_op_fuzz_gen_main",
    srcs = [
        "cc_op_fuzz_gen.cc",
        "cc_op_fuzz_gen.h",
        "cc_op_fuzz_gen_main.cc",
    ],
    copts = tf_copts(),
    data = [
        "//machina/core/api_def:base_api_def",
    ],
    deps = [
        "//machina/cc:cc_op_gen_util",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:op_gen_lib",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:hash",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:status",
    ],
)

# copybara:uncomment_begin
# tf_gen_op_wrappers_fuzz(
#     name = "array_ops_fuzz",
#     api_def_srcs = ["//machina/core/api_def:base_api_def"],
#     kernel_deps = [
#         "//machina/c/kernels:bitcast_op",
#         "//machina/core/kernels:array",
#         "//machina/core/kernels:check_numerics_op",
#         "//machina/core/kernels:fake_quant_ops",
#         "//machina/core/kernels:quantized_ops",
#         "//machina/core/kernels:scatter_nd_op",
#         "//machina/core/kernels/image:extract_image_patches_op",
#         "//machina/core/kernels/image:extract_volume_patches_op",
#         "//machina/core/kernels/image:mirror_pad_op",
#         "//machina/core/kernels/linalg:matrix_band_part_op",
#         "//machina/core/kernels/linalg:matrix_diag_op",
#         "//machina/core/kernels/linalg:matrix_set_diag_op",
#     ],
#     op_def_src = "//machina/core/ops:array_ops_op_lib",
# )
# copybara:uncomment_end

bzl_library(
    name = "op_fuzzing_bzl",
    srcs = ["op_fuzzing.bzl"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina:machina_bzl",
        "//machina/core/platform:rules_cc_bzl",
    ],
)
