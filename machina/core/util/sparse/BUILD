load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "higher_level_tests_group",
    srcs = ["sparse_tensor_test.cc"],
    visibility = ["//visibility:private"],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "dim_comparator.h",
        "group_iterator.cc",
        "group_iterator.h",
        "sparse_tensor.cc",
        "sparse_tensor.h",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "framework_group",
    srcs = [
        "group_iterator.h",
        "sparse_tensor.h",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "framework_internal_private_headers_group",
    srcs = [
        "dim_comparator.h",
        "group_iterator.h",
        "sparse_tensor.h",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "framework_internal_impl_group",
    srcs = [
        "group_iterator.cc",
        "sparse_tensor.cc",
    ],
    visibility = ["//machina/core:__pkg__"],
)

cc_library(
    name = "sparse_tensor",
    srcs = ["sparse_tensor.cc"],
    hdrs = [
        "dim_comparator.h",
        "sparse_tensor.h",
    ],
    deps = [
        "//machina/core:framework",
        "//machina/core:framework_lite",
        "//machina/core:framework_types_hdr",
        "//machina/core:lib",
        "//machina/core:lib_proto_parsing",
        "//machina/core:portable_gif_internal",
        "//machina/core/framework:bounds_check",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/lib/core:errors",
        "//machina/core/lib/core:status",
        "//machina/core/platform:errors",
        "@com_google_absl//absl/base:core_headers",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "sparse_tensor_test",
    size = "small",
    srcs = [
        "sparse_tensor_test.cc",
    ],
    deps = [
        ":sparse_tensor",
        "//machina/core:framework",
        "//machina/core:framework_lite",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core/platform:statusor",
        "@eigen_archive//:eigen3",
        "@local_xla//xla/tsl/platform:test_main",
    ],
)
