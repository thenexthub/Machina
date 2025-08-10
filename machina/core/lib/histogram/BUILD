load("//machina:machina.default.bzl", "filegroup")
load(
    "//machina/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        # machina/core:lib effectively exposes all targets under machina/core/lib/**
        "//machina/core:__pkg__",
        # machina/core/lib/monitoring:sampler uses histogram
        "//machina/core/lib/monitoring:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "histogram",
    hdrs = ["histogram.h"],
    deps = [
        "//machina/core/lib/gtl:array_slice",
        "//machina/core/platform:logging",
        "//machina/core/platform:macros",
        "//machina/core/platform:mutex",
        "//machina/core/platform:thread_annotations",
        "//machina/core/platform:types",
        "@local_xla//xla/tsl/lib/histogram",
    ],
    alwayslink = True,
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "histogram.h",
        "@local_xla//xla/tsl/lib/histogram:mobile_srcs_only_runtime",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_histogram_all_headers",
    srcs = [
        "histogram.h",
        "@local_xla//xla/tsl/lib/histogram:legacy_lib_histogram_all_headers",
    ],
    visibility = ["//machina/core:__pkg__"],
)
