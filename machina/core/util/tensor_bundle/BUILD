# Description:
#   Tensor bundle: a module to efficiently serialize and deserialize tensors.

load(
    "//machina:machina.bzl",
    "if_not_windows",
    "if_windows",
    "tf_cc_test",
    "tf_copts",
)
load("//machina:machina.default.bzl", "cc_header_only_library", "filegroup")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# To be exported to machina/core:mobile_srcs.
filegroup(
    name = "mobile_srcs",
    srcs = [
        "byte_swap_array.h",
        "byte_swap_tensor.cc",
        "byte_swap_tensor.h",
        "naming.cc",
        "naming.h",
        "tensor_bundle.cc",
        "tensor_bundle.h",
    ],
)

alias(
    name = "android_srcs",
    actual = ":mobile_srcs",
)

cc_library(
    name = "tensor_bundle",
    srcs = [
        "tensor_bundle.cc",
    ],
    hdrs = [
        "tensor_bundle.h",
    ],
    copts = tf_copts() + if_not_windows(["-Wno-sign-compare"]),
    linkopts = if_windows(["-DEFAULTLIB:ws2_32.lib"]),
    deps = [
        ":byteswaptensor",
        ":naming",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/lib/io:buffered_file",
        "@local_xla//xla/tsl/util:byte_swap_array",
    ],
)

cc_header_only_library(
    name = "tensor_bundle_headers_lib",
    features = ["-parse_headers"],  # Transitively pulls in Eigen headers
    deps = [":tensor_bundle"],
)

cc_library(
    name = "naming",
    srcs = ["naming.cc"],
    hdrs = ["naming.h"],
    deps = [
        "//machina/core:lib",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "byteswaparray",
    hdrs = ["byte_swap_array.h"],
    deps = [
        "//machina/core/platform:byte_order",
        "//machina/core/platform:errors",
        "//machina/core/platform:status",
        "@local_xla//xla/tsl/util:byte_swap_array",
    ],
)

cc_library(
    name = "byteswaptensor",
    srcs = ["byte_swap_tensor.cc"],
    hdrs = ["byte_swap_tensor.h"],
    deps = [
        ":byteswaparray",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:byte_order",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:platform_port",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:logging",
    ],
)

tf_cc_test(
    name = "tensor_bundle_test",
    srcs = ["tensor_bundle_test.cc"],
    data = ["//machina/core/util/tensor_bundle/testdata:old_string_tensors"],
    tags = [
        "nomsan",
        "notsan",
    ],
    deps = [
        ":byteswaptensor",
        ":naming",
        ":tensor_bundle",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/framework:tensor_testutil",
        "@com_google_absl//absl/status",
    ],
)
