load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.bzl", "tf_cc_test", "tf_features_nolayering_check_if_ios")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/lite:special_rules.bzl", "nonportable_visibility_allowlist")

# Kernel for custom parse_example
package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "parse_example",
    srcs = [
        "example_proto_fast_parsing.cc",
        "parse_example.cc",
    ],
    hdrs = [
        "example_proto_fast_parsing.h",
        "parse_example.h",
    ],
    compatible_with = get_compatible_with_portable(),
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        "//machina/core/platform:hash",
        "//machina/lite:framework",
        "//machina/lite:string_util",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels/internal:tensor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//machina:ios": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:core_cpu",
            "//machina/core:feature_util",
            "//machina/core:framework",
            "//machina/core:framework_internal",
            "//machina/core:lib",
            "//machina/core:lib_internal",
            "//machina/core:protos_all_cc",
        ],
    }),
)

tf_cc_test(
    name = "parse_example_test",
    srcs = ["parse_example_test.cc"],
    features = tf_features_nolayering_check_if_ios(),
    deps = [
        ":parse_example",
        "//machina/lite:framework",
        "//machina/lite:string_util",
        "//machina/lite/core:framework",
        "//machina/lite/core/api:op_resolver",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/kernels:test_main",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@flatbuffers",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//machina:ios": [
            "//machina/core:portable_machina_lib_lite",
        ],
        "//conditions:default": [
            "//machina/core:protos_all_cc",
            "//machina/core/example:feature_util",
            "//machina/core/platform:protobuf",
            "//machina/core/platform:tstring",
        ],
    }),
)

# Same as "parse_example" above, but doesn't use the "select" statements
# to use "portable_machina_lib_lite" for portable builds.
cc_library(
    name = "nonportable_parse_example",
    srcs = [
        "example_proto_fast_parsing.cc",
        "parse_example.cc",
    ],
    hdrs = [
        "example_proto_fast_parsing.h",
        "parse_example.h",
    ],
    visibility = nonportable_visibility_allowlist(),
    deps = [
        "//machina/core:core_cpu",
        "//machina/core:feature_util",
        "//machina/core:framework",
        "//machina/core:framework_internal",
        "//machina/core:lib",
        "//machina/core:lib_internal",
        "//machina/core:protos_all_cc",
        "//machina/lite:framework",
        "//machina/lite:string_util",
        "//machina/lite/core/c:common",
        "//machina/lite/kernels:kernel_util",
        "//machina/lite/kernels/internal:tensor",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "nonportable_parse_example_test",
    srcs = ["parse_example_test.cc"],
    deps = [
        ":nonportable_parse_example",
        "//machina/core:protos_all_cc",
        "//machina/core/example:feature_util",
        "//machina/core/platform:protobuf",
        "//machina/core/platform:tstring",
        "//machina/lite:framework",
        "//machina/lite:string_util",
        "//machina/lite/c:common",
        "//machina/lite/core:framework",
        "//machina/lite/core/api:op_resolver",
        "//machina/lite/core/c:common",
        "//machina/lite/core/kernels:builtin_ops",
        "//machina/lite/kernels:test_main",
        "//machina/lite/kernels:test_util",
        "//machina/lite/schema:schema_fbs",
        "@flatbuffers",
    ],
)
