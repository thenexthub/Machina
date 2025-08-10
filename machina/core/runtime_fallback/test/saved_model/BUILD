load("//machina:pytype.default.bzl", "pytype_strict_binary")
load("//machina:machina.bzl", "if_google", "tf_cc_shared_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [":internal"],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    packages = [
        "//machina/core/tfrt/saved_model/tests/...",
    ],
)

pytype_strict_binary(
    name = "gen_saved_model_v1",
    srcs = ["gen_saved_model_v1.py"],
    deps = [
        "//machina/python/client:session",
        "//machina/python/distribute:input_lib",  # build_cleaner: keep;
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/saved_model:builder",
        "//machina/python/saved_model:signature_constants",
        "//machina/python/saved_model:signature_def_utils",
        "//machina/python/saved_model:tag_constants",
        "//machina/python/saved_model:utils",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ] + if_google([
        "//learning/brain/public:disable_tf2",  # build_cleaner: keep; go/disable_tf2
    ]),
)

genrule(
    name = "saved_model_gen_saved_model_v1",
    srcs = [],
    outs = [
        "basic_v1/saved_model.pb",
    ],
    cmd = "$(location gen_saved_model_v1) --saved_model_path=$(RULEDIR)/basic_v1",
    tools = ["gen_saved_model_v1"],
)

tf_cc_shared_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.cc"],
    data = [
        "basic_v1/saved_model.pb",
    ],
    tags = ["no_oss"],
    deps = [
        "//machina/core/tfrt/saved_model:saved_model_testutil",
        "@com_google_googletest//:gtest_main",
    ] + select({
        "//machina:android": [
            "//machina/core:portable_machina_lib_lite",  # TODO(annarev): exclude runtime srcs
        ],
        "//conditions:default": [
            "//machina/cc/saved_model:loader",
            "//machina/core:test",
            "//machina/core/platform:resource_loader",
        ],
    }),
)
