# Fuzzing TensorFlow ops with GFT
# Most ops have a similar set of dependencies and a similar fuzzing
# infrastructure. Hence, we gather everything in one single place.
# Note that these fuzzers cover a large part of TF, they are not granular.

load(
    "//machina/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

# A trivial fuzzer with no pre-specified corpus.
tf_cc_fuzz_test(
    name = "identity_fuzz",
    srcs = ["identity_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/cc:cc_ops",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:array",
        "//machina/security/fuzzing/cc:fuzz_session",
        "//machina/security/fuzzing/cc/core/framework:tensor_domains",
        "//machina/security/fuzzing/cc/core/framework:tensor_shape_domains",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_fuzz_test(
    name = "concat_fuzz",
    srcs = ["concat_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/cc:cc_ops",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:array",
        "//machina/security/fuzzing/cc:fuzz_session",
        "//machina/security/fuzzing/cc/core/framework:tensor_domains",
        "//machina/security/fuzzing/cc/core/framework:tensor_shape_domains",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_fuzz_test(
    name = "add_fuzz",
    srcs = ["add_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/cc:cc_ops",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:array",
        "//machina/security/fuzzing/cc:fuzz_session",
        "//machina/security/fuzzing/cc/core/framework:datatype_domains",
        "//machina/security/fuzzing/cc/core/framework:tensor_domains",
        "//machina/security/fuzzing/cc/core/framework:tensor_shape_domains",
    ],
)

tf_cc_fuzz_test(
    name = "matmul_fuzz",
    srcs = ["matmul_fuzz.cc"],
    tags = [
        "no_oss",
        "noasan",  # b/283972985
    ],
    deps = [
        "//machina/cc:cc_ops",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:array",
        "//machina/core/kernels:matmul_op",
        "//machina/security/fuzzing/cc:fuzz_session",
        "//machina/security/fuzzing/cc/core/framework:tensor_domains",
    ],
)

tf_cc_fuzz_test(
    name = "bincount_fuzz",
    srcs = ["bincount_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/cc:cc_ops",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:array",
        "//machina/core/kernels:bincount_op",
        "//machina/security/fuzzing/cc:fuzz_session",
        "//machina/security/fuzzing/cc/core/framework:tensor_domains",
        "//machina/security/fuzzing/cc/core/framework:tensor_shape_domains",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_fuzz_test(
    name = "string_to_number_fuzz",
    srcs = ["string_to_number_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/cc:cc_ops",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:array",
        "//machina/core/kernels:string_to_number_op",
        "//machina/security/fuzzing/cc:fuzz_session",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_fuzz_test(
    name = "string_ops_fuzz",
    srcs = ["string_ops_fuzz.cc"],
    tags = ["no_oss"],
    deps = [
        "//machina/cc:cc_ops",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:array",
        "//machina/core/kernels:string",
        "//machina/core/kernels:string_split_op",
        "//machina/security/fuzzing/cc:fuzz_session",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_fuzz_test(
    name = "general_ops_fuzz",
    srcs = ["general_ops_fuzz.cc"],
    shard_count = 5,
    deps = [
        "//machina/cc:cc_ops",
        "//machina/core/framework:tensor",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/kernels:array",
        "//machina/core/kernels:decode_wav_op",
        "//machina/core/kernels/image",
        "//machina/core/kernels/image:decode_image_op",
        "//machina/core/ops:audio_ops_op_lib",
        "//machina/security/fuzzing/cc:fuzz_session",
        "//machina/security/fuzzing/cc/core/framework:tensor_domains",
    ],
)
