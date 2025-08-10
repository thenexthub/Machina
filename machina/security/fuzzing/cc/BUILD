# Fuzzing TensorFlow.

load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "//machina/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_cc_fuzz_test(
    name = "status_fuzz",
    srcs = ["status_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        ":fuzz_domains",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_fuzz_test(
    name = "arg_def_case_fuzz",
    srcs = ["arg_def_case_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/core/platform:str_util",
        "//machina/core/platform:stringpiece",
    ],
)

tf_cc_fuzz_test(
    name = "base64_fuzz",
    srcs = ["base64_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/core/platform:base64",
        "//machina/core/platform:status",
        "//machina/core/platform:stringpiece",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_fuzz_test(
    name = "bfloat16_fuzz",
    srcs = ["bfloat16_fuzz.cc"],
    tags = ["no_oss"],  # b/175698644
    deps = [
        "//machina/core:portable_gif_internal",
        "//machina/core/framework:bfloat16",
    ],
)

tf_cc_fuzz_test(
    name = "checkpoint_reader_fuzz",
    srcs = ["checkpoint_reader_fuzz.cc"],
    data = glob(["checkpoint_reader_testdata/*"]),
    tags = ["no_oss"],
    deps = [
        ":checkpoint_reader_fuzz_input_proto_cc",
        "//machina/c:checkpoint_reader",
        "//machina/c:tf_status",
        "//machina/c:tf_status_helper",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/platform:resource_loader",
        "//machina/core/util:saved_tensor_slice_proto_cc",
        "@local_xla//xla/tsl/platform:status",
    ],
)

tf_proto_library(
    name = "checkpoint_reader_fuzz_input_proto",
    srcs = ["checkpoint_reader_fuzz_input.proto"],
    make_default_target_header_only = True,
    protodeps = [
        "//machina/core/util:saved_tensor_slice_proto",
    ],
)

tf_cc_fuzz_test(
    name = "cleanpath_fuzz",
    srcs = ["cleanpath_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/core/platform:path",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_fuzz_test(
    name = "consume_leading_digits_fuzz",
    srcs = ["consume_leading_digits_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/core:portable_gif_internal",
        "//machina/core/platform:str_util",
        "//machina/core/platform:stringpiece",
    ],
)

tf_cc_fuzz_test(
    name = "joinpath_fuzz",
    srcs = ["joinpath_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/core/platform:path",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_fuzz_test(
    name = "status_group_fuzz",
    srcs = ["status_group_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        ":fuzz_domains",
        "//machina/core/platform:status",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "fuzz_domains",
    testonly = True,
    hdrs = ["fuzz_domains.h"],
    deps = [
        "//machina/core/platform:status",
        "@com_google_fuzztest//fuzztest",
    ],
)

tf_cc_fuzz_test(
    name = "stringprintf_fuzz",
    srcs = ["stringprintf_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/core/platform:stringprintf",
    ],
)

tf_cc_fuzz_test(
    name = "string_replace_fuzz",
    srcs = ["string_replace_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/core/platform:str_util",
        "//machina/core/platform:stringpiece",
    ],
)

tf_cc_fuzz_test(
    name = "tstring_fuzz",
    srcs = ["tstring_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/core/platform:tstring",
    ],
)

tf_cc_fuzz_test(
    name = "AreAttrValuesEqual_fuzz",
    srcs = ["AreAttrValuesEqual_fuzz.cc"],
    tags = ["no_oss"],  # b/175698644
    deps = [
        "//machina/core/framework:attr_value_proto_cc",
        "//machina/core/framework:attr_value_util",
    ],
)

tf_cc_fuzz_test(
    name = "ParseAttrValue_fuzz",
    srcs = ["ParseAttrValue_fuzz.cc"],
    tags = ["no_oss"],  # b/175698644
    deps = [
        "//machina/core/framework:attr_value_proto_cc",
        "//machina/core/framework:attr_value_util",
        "//machina/core/platform:stringpiece",
    ],
)

tf_cc_fuzz_test(
    name = "parseURI_fuzz",
    srcs = ["parseURI_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/core/platform:path",
        "//machina/core/platform:stringpiece",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fuzz_session",
    testonly = 1,
    hdrs = ["fuzz_session.h"],
    tags = ["no_oss"],
    visibility = [
        "//machina/cc/framework/fuzzing:__subpackages__",
        "//machina/security/fuzzing:__subpackages__",
    ],
    deps = [
        "//machina/cc:scope",
        "//machina/core:core_cpu_base",
        "//machina/core:session_options",
        "//machina/core/common_runtime:direct_session_internal",
        "//machina/core/framework:tensor",
        "//machina/core/platform:status",
        "@com_google_fuzztest//fuzztest",
    ],
)

tf_cc_fuzz_test(
    name = "end_to_end_fuzz",
    srcs = ["end_to_end_fuzz.cc"],
    deps = [
        "//machina/cc/saved_model:constants",
        "//machina/cc/saved_model:loader",
        "//machina/cc/saved_model:tag_constants",
        "//machina/core:core_cpu",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:machina",
        "//machina/core/framework:tensor",
        "//machina/core/platform:status",
        "//machina/security/fuzzing/cc/core/framework:datatype_domains",
        "//machina/security/fuzzing/cc/core/framework:tensor_domains",
        "//machina/security/fuzzing/cc/core/framework:tensor_shape_domains",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:status",
    ],
)

tf_cc_fuzz_test(
    name = "text_literal_reader_fuzz",
    srcs = ["text_literal_reader_fuzz.cc"],
    deps = [
        "@local_xla//xla:text_literal_reader",
        "@local_xla//xla/hlo/parser:hlo_parser",
        "@local_xla//xla/tsl/platform:env",
    ],
)
