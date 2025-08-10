load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "filegroup", "tf_gen_op_libs", "tf_kernel_library")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_kernel_library(
    name = "bitcast_op",
    prefix = "bitcast_op",
    deps = [
        "//machina/c:kernels",
        "//machina/c:ops",
        "//machina/c:tf_datatype",
        "//machina/c:tf_status",
        "//machina/c:tf_tensor",
        "//machina/core:framework",
        "//machina/core:lib",
        "@com_google_absl//absl/log:check",
    ],
)

tf_kernel_library(
    name = "summary_op",
    prefix = "summary_op",
    deps = [
        "//machina/c:kernels",
        "//machina/c:tf_status",
        "//machina/c:tf_tensor",
        "//machina/c/kernels:tensor_shape_utils",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "histogram_summary_op",
    prefix = "histogram_summary_op",
    deps = [
        "//machina/c:kernels",
        "//machina/c:tf_status",
        "//machina/c:tf_tensor",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/log:check",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "merge_summary_op",
    prefix = "merge_summary_op",
    deps = [
        "//machina/c:kernels",
        "//machina/c:tf_status",
        "//machina/c:tf_tensor",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/log:check",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["bitcast"],
    deps = [
        "//machina/c:ops",
        "//machina/c:tf_datatype",
        "//machina/c:tf_status",
        "//machina/c:tf_tensor",
        "//machina/core:lib",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["summary"],
    deps = [
        "//machina/c:ops",
        "//machina/c:tf_status",
        "//machina/core:lib",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["histogram_summary"],
    deps = [
        "//machina/c:ops",
        "//machina/c:tf_status",
        "//machina/core:lib",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["merge_summary"],
    deps = [
        "//machina/c:ops",
        "//machina/c:tf_status",
        "//machina/core:lib",
    ],
)

tf_cc_test(
    name = "bitcast_op_test",
    srcs = ["bitcast_op_test.cc"],
    deps = [
        ":bitcast_op",
        "//machina/core:framework",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

tf_cc_test(
    name = "summary_op_test",
    srcs = ["summary_op_test.cc"],
    deps = [
        ":summary_op",
        "//machina/c:kernels",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

tf_cc_test(
    name = "summary_op_benchmark_test",
    size = "small",
    srcs = ["summary_op_benchmark_test.cc"],
    deps = [
        ":summary_op",
        "//machina/c:kernels",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
    ],
)

cc_library(
    name = "tensor_shape_utils",
    srcs = ["tensor_shape_utils.cc"],
    hdrs = ["tensor_shape_utils.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//machina/c:tf_tensor",
        "//machina/core:lib",
    ],
)

tf_cc_test(
    name = "tensor_shape_utils_test",
    srcs = ["tensor_shape_utils_test.cc"],
    deps = [
        ":tensor_shape_utils",
        "//machina/c:tf_tensor_internal",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)

# Be aware that this target is included by
# //third_party/machina/core/kernels:portable_core_ops
#
filegroup(
    name = "portable_all_op_kernels",
    srcs = [
        "bitcast_op.cc",
        "histogram_summary_op.cc",
        "merge_summary_op.cc",
        "summary_op.cc",
        "tensor_shape_utils.cc",
        "tensor_shape_utils.h",
    ],
)

filegroup(
    name = "portable_all_ops",
    srcs = [
        "ops/bitcast.cc",
        "ops/histogram_summary.cc",
        "ops/merge_summary.cc",
        "ops/summary.cc",
    ],
)
