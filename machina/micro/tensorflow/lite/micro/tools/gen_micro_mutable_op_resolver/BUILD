load("@rules_python//python:defs.bzl", "py_binary")
load("@tflm_pip_deps//:requirements.bzl", "requirement")

package(
    default_visibility = ["//:__subpackages__"],
    licenses = ["notice"],
)

py_binary(
    name = "generate_micro_mutable_op_resolver_from_model",
    srcs = [
        "generate_micro_mutable_op_resolver_from_model.py",
    ],
    data = [
        "templates/micro_mutable_op_resolver.h.mako",
    ],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        requirement("mako"),
        "//machina/lite/python:schema_py",
        "//machina/lite/python:schema_util",
        "//machina/lite/tools:flatbuffer_utils",
        "//machina/lite/tools:visualize",
    ],
)

py_binary(
    name = "generate_micro_mutable_op_resolver_from_model_test",
    srcs = [
        "generate_micro_mutable_op_resolver_from_model_test.py",
    ],
    data = [
        "templates/BUILD.mako",
        "templates/micro_mutable_op_resolver.h.mako",
        "templates/micro_mutable_op_resolver_test.cc.mako",
    ],
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        requirement("machina"),
        requirement("mako"),
        "//machina/lite/micro/tools:generate_test_for_model",
        "//machina/lite/python:schema_py",
        "//machina/lite/python:schema_util",
        "//machina/lite/tools:flatbuffer_utils",
    ],
)
