load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.default.bzl", "tf_gen_op_libs", "tf_gen_op_wrapper_cc")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/core/tfrt/__subpackages__"],
)

tf_gen_op_libs(
    op_lib_names = ["ifrt_program_ops"],
    sub_directory = "",
    deps = ["//machina/core:lib"],
)

tf_gen_op_wrapper_cc(
    name = "gen_ifrt_program_ops",
    out_ops_file = "gen_ifrt_program_ops",
    deps = [":ifrt_program_ops_op_lib"],
)

cc_library(
    name = "gen_ifrt_program_ops_cc",
    srcs = ["gen_ifrt_program_ops.cc"],
    hdrs = ["gen_ifrt_program_ops.h"],
    deps = [
        ":ifrt_program_ops_op_lib",
        "//machina/cc:const_op",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/core:framework",
        "//machina/core:lib",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["stream_ops"],
    sub_directory = "",
    deps = ["//machina/core:lib"],
)

tf_gen_op_wrapper_cc(
    name = "gen_stream_ops",
    out_ops_file = "gen_stream_ops",
    deps = [":stream_ops_op_lib"],
)

cc_library(
    name = "gen_stream_ops_cc",
    srcs = ["gen_stream_ops.cc"],
    hdrs = ["gen_stream_ops.h"],
    deps = [
        ":stream_ops_op_lib",
        "//machina/cc:const_op",
        "//machina/cc:ops",
        "//machina/cc:scope",
        "//machina/core:framework",
        "//machina/core:lib",
    ],
)
