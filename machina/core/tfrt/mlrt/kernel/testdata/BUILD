load("//machina:strict.default.bzl", "py_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/core/tfrt/mlrt/kernel:__subpackages__",
    ],
)

filegroup(
    name = "testdata",
    srcs = [
        "gen_checkpoint_data/saved_model.pb",
        "gen_checkpoint_data/variables/variables.data-00000-of-00001",
        "gen_checkpoint_data/variables/variables.index",
    ],
    visibility = ["//visibility:public"],
)

py_strict_binary(
    name = "gen_checkpoint",
    srcs = ["gen_checkpoint.py"],
    deps = [
        "//machina/python/compat:v2_compat",
        "//machina/python/eager/polymorphic_function",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/module",
        "//machina/python/ops:variables",
        "//machina/python/saved_model",
        "//machina/python/saved_model:save_options",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@absl_py//absl/logging",
    ],
)

genrule(
    name = "checkpoint_data",
    srcs = [],
    outs = [
        "gen_checkpoint_data/variables/variables.data-00000-of-00001",
        "gen_checkpoint_data/variables/variables.index",
        "gen_checkpoint_data/saved_model.pb",
    ],
    cmd = "$(location gen_checkpoint)" + " --saved_model_path=$(RULEDIR)/gen_checkpoint_data",
    tools = ["gen_checkpoint"],
)
