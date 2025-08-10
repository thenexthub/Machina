# Description:
# wav_io targets.

load(
    "//machina/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/core:__pkg__"],
    licenses = ["notice"],
)

exports_files(srcs = [
    "wav_io.cc",
    "wav_io.h",
    "wav_io_test.cc",
])

cc_library(
    name = "wav_io",
    srcs = [
        "wav_io.cc",
    ],
    hdrs = ["wav_io.h"],
    deps = [
        "//machina/core/lib/core:coding",
        "//machina/core/lib/core:errors",
        "//machina/core/lib/core:status",
        "//machina/core/platform:byte_order",
        "//machina/core/platform:logging",
        "//machina/core/platform:macros",
        "//machina/core/platform:types",
        "@com_google_absl//absl/base",
    ],
)
