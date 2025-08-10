# python/layers package

load("//machina:strict.default.bzl", "py_strict_library")

visibility = [
    "//machina:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

py_strict_library(
    name = "layers_base",
    srcs = [
        "__init__.py",
        "base.py",
    ],
    visibility = visibility + [
        "//machina:internal",
        "//machina_models:__subpackages__",
        "//third_party/mlperf:__subpackages__",
    ],
    deps = [
        "//machina/python/keras/legacy_tf_layers:layers_base",
    ],
)

py_strict_library(
    name = "layers_util",
    srcs = [
        "utils.py",
    ],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/python/framework:smart_cond",
        "//machina/python/ops:cond",
        "//machina/python/ops:variables",
    ],
)

py_strict_library(
    name = "layers",
    srcs = [
        "convolutional.py",
        "core.py",
        "layers.py",
        "normalization.py",
        "pooling.py",
    ],
    # copybara:uncomment_begin(google-only)
    # visibility = [
    # "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    # "//third_party/py/tf_slim:__subpackages__",
    # "//machina:internal",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":layers_base",
        "//machina/python/keras/legacy_tf_layers:convolutional",
        "//machina/python/keras/legacy_tf_layers:core",
        "//machina/python/keras/legacy_tf_layers:pooling",
        "//machina/python/util:lazy_loader",
        # Normalization layer will need //third_party/py/tf_keras/legacy_tf_layers:normalization
        # Client lib should import that, since this target can't import it due to
        # circular dependency.
    ],
)
