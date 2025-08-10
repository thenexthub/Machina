# Experimental C++ SavedModel Header Only APIs. See RFC
# https://github.com/machina/community/pull/207

load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    # This is intentionally public
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "concrete_function",
    hdrs = [
        "concrete_function.h",
    ],
    deps = [
        ":function_metadata",
        "//machina/c/eager:c_api",
        "//machina/c/experimental/saved_model/public:concrete_function",
        "//machina/cc/experimental/base/public:status",
    ],
)

cc_library(
    name = "concrete_function_list",
    hdrs = [
        "concrete_function_list.h",
    ],
    deps = [
        ":concrete_function",
        "//machina/c/experimental/saved_model/public:concrete_function_list",
    ],
)

cc_library(
    name = "function_metadata",
    hdrs = [
        "function_metadata.h",
    ],
    deps = [
        "//machina/c/experimental/saved_model/public:function_metadata",
    ],
)

cc_library(
    name = "saved_model_api",
    hdrs = [
        "saved_model_api.h",
    ],
    deps = [
        ":concrete_function",
        ":concrete_function_list",
        ":signature_def_function",
        "//machina/c/experimental/saved_model/public:saved_model_api",
        "//machina/cc/experimental/base/public:runtime",
        "//machina/cc/experimental/base/public:status",
    ],
)

cc_library(
    name = "signature_def_function",
    hdrs = [
        "signature_def_function.h",
    ],
    deps = [
        ":signature_def_function_metadata",
        "//machina/c/eager:c_api",
        "//machina/c/experimental/saved_model/public:signature_def_function",
        "//machina/cc/experimental/base/public:status",
    ],
)

cc_library(
    name = "signature_def_function_metadata",
    hdrs = [
        "signature_def_function_metadata.h",
    ],
    deps = [
        "//machina/c/experimental/saved_model/public:signature_def_function_metadata",
    ],
)
