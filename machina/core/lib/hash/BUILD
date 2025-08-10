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
        # machina/core/lib/gtl uses hash functions as part of their flatset/map representation
        "//machina/core/lib/gtl:__pkg__",
        # machina/core/lib/io/table_builder.cc uses crc functionality
        "//machina/core/lib/io:__pkg__",
        # machina/core/framework:tensor depends on hash
        "//machina/core/framework:__pkg__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "crc32c",
    hdrs = ["crc32c.h"],
    deps = [
        "//machina/core/platform",
        "//machina/core/platform:cord",
        "//machina/core/platform:types",
        "@local_xla//xla/tsl/lib/hash:crc32c",
    ],
)

cc_library(
    name = "hash",
    hdrs = ["hash.h"],
    deps = [
        "//machina/core/platform:hash",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "hash.h",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "crc32c.h",
        "@local_xla//xla/tsl/lib/hash:mobile_srcs_only_runtime",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_hash_all_headers",
    srcs = [
        "crc32c.h",
        "hash.h",
        "@local_xla//xla/tsl/lib/hash:legacy_lib_hash_all_headers",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_public_headers",
    srcs = [
        "hash.h",
    ],
    visibility = ["//machina/core:__pkg__"],
)
