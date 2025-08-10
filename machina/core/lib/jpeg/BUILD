# Description:
# JPEG handle and mem packages.

load(
    "//machina:machina.bzl",
    "if_android",
    "if_mobile",
    "tf_cc_test",
    "tf_copts",
)
load(
    "//machina/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/core:__pkg__"],
    features = ["-parse_headers"],
    licenses = ["notice"],
)

cc_library(
    name = "jpeg_internal",
    srcs = [
        "jpeg_handle.cc",
        "jpeg_mem.cc",
        "//machina/core/platform:jpeg_hdrs",
    ],
    hdrs = [
        "jpeg_handle.h",
        "jpeg_mem.h",
    ],
    copts = tf_copts(),
    linkopts = select({
        "//machina:freebsd": [],
        "//machina:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/platform:jpeg",
        "@libjpeg_turbo//:jpeg",
    ],
)

cc_library(
    name = "portable_jpeg_internal",
    srcs = if_mobile([
        "jpeg_handle.cc",
        "jpeg_mem.cc",
        "//machina/core/platform:jpeg_hdrs",
    ]),
    hdrs = [
        "jpeg_handle.h",
        "jpeg_mem.h",
        "//machina/core/lib/core:legacy_lib_core_stringpiece_header",
        "//machina/core/platform:jpeg_internal_hdrs",
        "@local_tsl//tsl/platform:jpeg_internal_hdrs",
        "@local_xla//xla/tsl/platform/default:integral_types.h",
        "@local_xla//xla/tsl/platform/default:logging.h",
    ],
    copts = tf_copts(),
    linkopts = if_android(["-ldl"]),
    deps = [
        "//machina/core:core_stringpiece",
        "//machina/core/platform:byte_order",
        "//machina/core/platform:cord",
        "//machina/core/platform:dynamic_annotations",
        "//machina/core/platform:float8",
        "//machina/core/platform:jpeg",
        "//machina/core/platform:logging",
        "//machina/core/platform:stringpiece",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
        "@libjpeg_turbo//:jpeg",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:tstring",
        "@local_tsl//tsl/platform:types",
        "@local_xla//xla/tsl/platform:types",
    ],
)

tf_cc_test(
    name = "lib_jpeg_jpeg_mem_unittest",
    srcs = ["jpeg_mem_unittest.cc"],
    data = ["//machina/core/lib/jpeg/testdata"],
    deps = [
        ":jpeg_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:jpeg",
        "@com_google_absl//absl/base",
        "@libjpeg_turbo//:jpeg",
    ],
)
