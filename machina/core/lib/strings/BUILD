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
        # machina/core/framework references some of the headers in this package
        "//machina/core/framework:__subpackages__",
        "//machina/core/tpu:__subpackages__",
        # For proto_serialization library
        "//machina/cc/experimental/tf2:__subpackages__",
        #For ordered_code library
        "//machina/python/profiler/internal:__pkg__",
    ],
    licenses = ["notice"],
)

# Todo(bmzhao): Remaining targets to add to this BUILD file are: all tests.

cc_library(
    name = "base64",
    hdrs = ["base64.h"],
    deps = [
        "//machina/core/platform:base64",
    ],
)

cc_library(
    name = "numbers",
    hdrs = ["numbers.h"],
    deps = ["//machina/core/platform:numbers"],
)

cc_library(
    name = "ordered_code",
    srcs = ["ordered_code.cc"],
    hdrs = ["ordered_code.h"],
    deps = [
        "//machina/core/platform:logging",
        "//machina/core/platform:macros",
        "//machina/core/platform:stringpiece",
        "//machina/core/platform:types",
        "@com_google_absl//absl/log:check",
        "@local_xla//xla/tsl/lib/core:bits",
    ],
)

cc_library(
    name = "proto_serialization",
    hdrs = ["proto_serialization.h"],
    deps = [
        "@local_xla//xla/tsl/lib/strings:proto_serialization",
    ],
)

cc_library(
    name = "proto_text_util",
    srcs = ["proto_text_util.cc"],
    hdrs = ["proto_text_util.h"],
    deps = [
        "//machina/core/platform:numbers",
        "//machina/core/platform:scanner",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "scanner",
    hdrs = ["scanner.h"],
    deps = ["//machina/core/platform:scanner"],
)

cc_library(
    name = "str_util",
    hdrs = ["str_util.h"],
    deps = ["//machina/core/platform:str_util"],
)

cc_library(
    name = "strcat",
    hdrs = ["strcat.h"],
    deps = ["//machina/core/platform:strcat"],
)

cc_library(
    name = "stringprintf",
    hdrs = ["stringprintf.h"],
    deps = ["//machina/core/platform:stringprintf"],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "proto_text_util.cc",
        "proto_text_util.h",
        "scanner.h",
        "str_util.h",
        "strcat.h",
        "stringprintf.h",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "base64.h",
        "numbers.h",
        "ordered_code.cc",
        "ordered_code.h",
        "proto_serialization.h",
        "@local_xla//xla/tsl/lib/strings:mobile_srcs_only_runtime",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_strings_all_headers",
    srcs = [
        "base64.h",
        "numbers.h",
        "ordered_code.h",
        "proto_serialization.h",
        "proto_text_util.h",
        "scanner.h",
        "str_util.h",
        "strcat.h",
        "stringprintf.h",
        "@local_xla//xla/tsl/lib/strings:legacy_lib_strings_all_headers",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_strings_all_tests",
    srcs = [
        "base64_test.cc",
        "ordered_code_test.cc",
        "proto_serialization_test.cc",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_proto_parsing_headers",
    srcs = [
        "numbers.h",
        "strcat.h",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_string_headers",
    srcs = [
        "numbers.h",
        "proto_serialization.h",
        "str_util.h",
        "strcat.h",
        "stringprintf.h",
        "@local_xla//xla/tsl/lib/strings:legacy_lib_string_headers",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_public_string_headers",
    srcs = [
        "base64.h",
        "ordered_code.h",
        "proto_serialization.h",
        "proto_text_util.h",
        "scanner.h",
        "@local_xla//xla/tsl/lib/strings:legacy_lib_internal_public_string_headers",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "legacy_low_level_library_tests",
    srcs = [
        "base64_test.cc",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "legacy_strings_ordered_code_test",
    srcs = [
        "ordered_code_test.cc",
    ],
    visibility = ["//machina/core:__pkg__"],
)

filegroup(
    name = "legacy_strings_proto_serialization_test",
    srcs = [
        "proto_serialization_test.cc",
    ],
    visibility = ["//machina/core:__pkg__"],
)
