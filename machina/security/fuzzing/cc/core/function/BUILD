# Fuzzing TensorFlow/core/function with GFT

load(
    "//machina/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

tf_cc_fuzz_test(
    name = "runtime_client_fuzz",
    srcs = ["runtime_client_fuzz.cc"],
    deps = [
        "//machina/c/eager:immediate_execution_context",
        "//machina/core:core_cpu",
        "//machina/core:framework",
        "//machina/core:framework_types_hdr",
        "//machina/core:portable_gif_internal",
        "//machina/core/common_runtime/eager:context",
        "//machina/core/framework:function_proto_cc",
        "//machina/core/framework:tensor",
        "//machina/core/framework:types_proto_cc",
        "//machina/core/function/runtime_client:runtime_client_cc",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:status",
    ],
)
