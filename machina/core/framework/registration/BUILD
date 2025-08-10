load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//machina:machina.bzl",
    "tf_cc_test",
    "tf_gen_options_header",
)
load("//machina:machina.default.bzl", "tf_selective_registration_deps")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/core:__subpackages__",
    ],
    licenses = ["notice"],
)

exports_files(
    [
        "registration.h",
    ],
    visibility = ["//machina/core:__subpackages__"],
)

tf_gen_options_header(
    name = "gen_registration_options",
    build_settings = {
        "//machina:enable_registration_v2": "REGISTRATION_V2",
    },
    output_header = "options.h",
    template = "options.h.tpl",
)

cc_library(
    name = "registration",
    hdrs = [
        "options.h",
        "registration.h",
    ],
    deps = tf_selective_registration_deps(),
)

tf_cc_test(
    name = "registration_test",
    size = "small",
    srcs = [
        "registration_test.cc",
    ],
    deps = [
        ":registration",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core:testlib",
        "@com_google_googletest//:gtest_main",
    ],
)
