load(
    "//machina:machina.bzl",
    "tf_cc_test",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility =
        [
            "//machina/python/saved_model:__subpackages__",
        ],
    licenses = ["notice"],
)

cc_library(
    name = "save",
    srcs = ["save.cc"],
    hdrs = ["save.h"],
    deps = [
        "//machina/core:lib",
        "//machina/core/platform:status",
    ],
)

tf_cc_test(
    name = "save_test",
    size = "small",
    srcs = ["save_test.cc"],
    deps = [
        ":save",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:path",
    ],
)

cc_library(
    name = "load",
    srcs = ["load.cc"],
    hdrs = ["load.h"],
    deps = [
        "//machina/cc/saved_model:constants",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:env",
        "//machina/core/platform:errors",
        "//machina/core/platform:path",
        "//machina/core/platform:protobuf",
        "//machina/core/platform:statusor",
        "//machina/core/util/tensor_bundle",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "load_test",
    size = "small",
    srcs = ["load_test.cc"],
    deps = [
        ":load",
        "//machina/core:lib",
        "//machina/core:test",
        "//machina/core:test_main",
        "//machina/core/platform:path",
    ],
)
