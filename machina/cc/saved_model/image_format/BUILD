# On-disk serialization format for TensorFlow models.

load(
    "//machina:machina.bzl",
    "if_not_windows_or_mac",
)
load("//machina/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//smartass/brain/configure:__subpackages__",
        "//machina/cc/experimental/tfa:__subpackages__",
        "//machina/cc/saved_model:__subpackages__",
        "//machina/tools/tfg_graph_transforms:__subpackages__",
        "//waymo/onboard/ml/chauffeur_net/utils:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "internal_api",
    srcs = ["internal_api.cc"],
    hdrs = ["internal_api.h"],
    deps = [
        "//machina/cc/saved_model:metrics",
        "//machina/cc/saved_model:util",
        "//machina/core/platform",
        "//machina/core/platform:env",
        "//machina/core/protobuf:for_core_protos_cc",
        "//machina/tools/proto_splitter/cc:max_size",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ] + if_not_windows_or_mac([
        "//machina/tools/proto_splitter:merge",
        "//machina/tools/proto_splitter/cc:saved_model_splitter",
    ]),
)
