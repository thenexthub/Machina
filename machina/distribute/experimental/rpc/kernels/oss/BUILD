load("@local_xla//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//machina:machina.default.bzl", "tf_grpc_cc_dependencies")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina/distribute/experimental/rpc/kernels:__subpackages__"],
    licenses = ["notice"],
)

cc_library(
    name = "grpc_credentials",
    srcs = ["grpc_credentials.cc"],
    deps = tf_grpc_cc_dependencies(),
    alwayslink = 1,
)
