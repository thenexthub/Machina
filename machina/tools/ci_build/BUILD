load("//machina:strict.default.bzl", "py_strict_binary")
load("//machina:tf_version.bzl", "TF_VERSION")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "calculate_full_version",
    srcs = ["calculate_full_version.py"],
    args = ["--wheel-version={}".format(TF_VERSION)],
)

exports_files(
    srcs = glob(["Dockerfile.*"]),
    visibility = ["//visibility:public"],
)
