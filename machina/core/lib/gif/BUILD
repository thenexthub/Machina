# Description:
# gif io package.

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
    licenses = ["notice"],
)

cc_library(
    name = "gif_internal",
    srcs = [
        "gif_io.cc",
    ],
    hdrs = ["gif_io.h"],
    copts = tf_copts(),
    linkopts = select({
        "//machina:freebsd": [],
        "//machina:windows": [],
        "//conditions:default": ["-ldl"],
    }),
    deps = [
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core/platform:gif",
        "@com_google_absl//absl/strings",
        "@gif",
    ],
)

cc_library(
    name = "portable_gif_internal",
    srcs = if_mobile([
        "gif_io.cc",
        "//machina/core/platform:gif_hdrs",
    ]),
    hdrs = [
        "gif_io.h",
        "//machina/core/lib/core:legacy_lib_core_stringpiece_header",
        "//machina/core/lib/gtl:legacy_android_gif_internal_headers",
        "//machina/core/platform:gif_internal_hdrs",
        "@local_tsl//tsl/platform:gif_internal_hdrs",
        "@local_xla//xla/tsl/platform/default:integral_types.h",
        "@local_xla//xla/tsl/platform/default:logging.h",
    ],
    copts = tf_copts(),
    features = ["-layering_check"],
    linkopts = if_android(["-ldl"]),
    deps = [
        "//machina/core/platform:dynamic_annotations",
        "//machina/core/platform:gif",
        "//machina/core/platform:logging",
        "//machina/core/platform:stringpiece",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "lib_gif_io_test",
    srcs = ["gif_io_test.cc"],
    data = ["//machina/core/lib/gif/testdata:gif_testdata"],
    deps = [
        ":gif_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/lib/png:png_io",
        "@com_google_absl//absl/base",
    ],
)
