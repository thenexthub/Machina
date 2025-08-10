load("//machina:machina.bzl", "tf_cc_test")
load("//machina:machina.default.bzl", "filegroup")
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "graph_verifier.h",
    ],
    visibility = [
        "//machina/python/grappler:__pkg__",
    ],
)

cc_library(
    name = "graph_verifier",
    hdrs = [
        "graph_verifier.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
    ],
)

cc_library(
    name = "structure_verifier",
    srcs = ["structure_verifier.cc"],
    hdrs = [
        "structure_verifier.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_verifier",
        "//machina/core:framework",
        "//machina/core:graph",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core/grappler/utils:topological_sort",
    ],
)

tf_cc_test(
    name = "structure_verifier_test",
    srcs = ["structure_verifier_test.cc"],
    deps = [
        ":structure_verifier",
        "//machina/cc:cc_ops",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core:protos_all_cc",
        "//machina/core:test",
        "//machina/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)
