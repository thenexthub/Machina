load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

# Provides stable ABI delegate.
load("//machina/lite:build_def.bzl", "tflite_copts")

# copybara:uncomment package(default_applicable_licenses = ["//machina:license"])

cc_library(
    name = "delegate_provider",
    srcs = [
        "stable_delegate_provider.cc",
    ],
    copts = tflite_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//machina/lite/c:common",
        "//machina/lite/tools:command_line_flags",
        "//machina/lite/tools:logging",
        "//machina/lite/tools:tool_params",
        "//machina/lite/tools/delegates:delegate_provider_hdr",
    ] + select({
        # Stable delegate does not support Windows because the shared library loader hasn't been
        # extended to support Windows.
        "//machina:windows": [],
        "//conditions:default": [
            "//machina/lite/acceleration/configuration:configuration_fbs",
            "//machina/lite/acceleration/configuration/c:delegate_plugin",
            "//machina/lite/acceleration/configuration/c:stable_delegate",
            "//machina/lite/delegates/utils/experimental/stable_delegate:delegate_loader",
            "//machina/lite/delegates/utils/experimental/stable_delegate:tflite_settings_json_parser",
        ],
    }),
    # Statically registers itself with DelegateProviderRegistrar.
    alwayslink = 1,
)

cc_test(
    name = "delegate_provider_test",
    size = "small",
    srcs = ["stable_delegate_provider_test.cc"],
    data = [
        ":test_invalid_settings.json",
        ":test_missing_delegate_path_settings.json",
        ":test_missing_stable_delegate_settings.json",
        ":test_sample_stable_delegate_settings.json",
        ":test_stable_xnnpack_settings.json",
        "//machina/lite/delegates/utils/experimental/sample_stable_delegate:machinalite_sample_stable_delegate",
        "//machina/lite/delegates/utils/experimental/stable_delegate:machinalite_stable_xnnpack_delegate",
    ],
    # Disable the test on Windows as the shared library loader doesn't support it.
    tags = ["no-windows"],
    deps = [
        ":delegate_provider",
        "//machina/lite/delegates/xnnpack:xnnpack_delegate",
        "//machina/lite/tools:tool_params",
        "//machina/lite/tools/delegates:delegate_provider_hdr",
        "//machina/lite/tools/delegates:delegate_provider_lib",
        "@com_google_googletest//:gtest_main",
        "@pthreadpool",
    ],
)

exports_files(
    srcs = [
        "test_invalid_settings.json",
        "test_sample_stable_delegate_settings.json",
    ],
    visibility = ["//visibility:public"],
)
