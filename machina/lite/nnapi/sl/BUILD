load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//machina:machina.default.bzl", "get_compatible_with_portable")
load("//machina/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//machina/lite:special_rules.bzl", "nnapi_sl_visibility_allowlist")

_DEFAULT_VISIBILITY = ["//machina/lite:__subpackages__"]

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = _DEFAULT_VISIBILITY,
    licenses = ["notice"],
)

cc_library(
    name = "nnapi_support_library_headers",
    hdrs = [
        "include/SupportLibrary.h",
        "include/SupportLibrarySymbols.h",
        "public/NeuralNetworksSupportLibraryImpl.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = _DEFAULT_VISIBILITY + nnapi_sl_visibility_allowlist(),
    deps = [
        "//machina/lite/kernels/internal:compatibility",
        "//machina/lite/nnapi:nnapi_lib",
    ],
)

cc_library(
    name = "nnapi_support_library",
    srcs = [
        "SupportLibrary.cc",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = _DEFAULT_VISIBILITY + nnapi_sl_visibility_allowlist(),
    deps = [
        ":nnapi_support_library_headers",
        "//machina/lite:minimal_logging",
        "//machina/lite/nnapi:nnapi_lib",
    ],
)
