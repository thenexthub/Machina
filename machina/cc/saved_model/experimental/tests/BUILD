# Tests for the C++ header-only SavedModelAPI.
load("//machina:machina.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

tf_cc_test(
    name = "saved_model_api_test",
    srcs = [
        "saved_model_api_test.cc",
    ],
    data = [
        "//machina/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        "//machina/c:tf_status_headers",
        "//machina/cc/experimental/base/public:runtime",
        "//machina/cc/experimental/base/public:runtime_builder",
        "//machina/cc/experimental/base/public:status",
        "//machina/cc/saved_model/experimental/public:saved_model_api",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
    ],
)
