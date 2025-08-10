# Description:
# Go API for TensorFlow.

load(
    "//machina:machina.bzl",
    "tf_shared_library_deps",
)

package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

sh_test(
    name = "test",
    size = "small",
    srcs = ["test.sh"],
    data = [
        ":all_files",  # Go sources
        "//machina/c:headers",  # C library header
        "//machina/c/eager:headers",  # Eager C library header
        "//machina/cc/saved_model:saved_model_half_plus_two",  # Testdata for LoadSavedModel
    ] + tf_shared_library_deps(),
    # TODO: Enable this test again once protos are supported by bazel.
    tags = ["manual"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//machina:__subpackages__"],
)
