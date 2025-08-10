# Description:
# CLIF wrappers for TensorFlow SavedModels.

load("//machina/core/platform:build_config.bzl", "tf_py_clif_cc")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

tf_py_clif_cc(
    name = "loader",
    srcs = ["loader.clif"],
    deps = [
        "//machina/cc/saved_model:loader",
    ],
)
