load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:strict.default.bzl", "py_strict_test")

# Placeholder: load py_proto_library
load(
    "//machina/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "trace_type",
    srcs = [
        "__init__.py",
        "trace_type_builder.py",
    ],
    visibility = ["//machina:internal"],
    deps = [
        ":custom_nest_trace_type",
        ":default_types",
        ":serialization",
        ":util",
        "//machina/python/types:trace",
        "//machina/python/util:custom_nest_protocol",
    ],
)

pytype_strict_library(
    name = "util",
    srcs = [
        "util.py",
    ],
    visibility = ["//machina:internal"],
    deps = [
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "trace_type_test",
    srcs = ["trace_type_test.py"],
    # TODO(b/213375459): For continous benchmark monitoring
    visibility = ["//learning/brain/contrib/eager/python/examples:__pkg__"],
    deps = [
        ":custom_nest_trace_type",
        ":default_types",
        ":trace_type",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina/python/compat:v2_compat",
        "//machina/python/data/ops:dataset_ops",
        "//machina/python/data/ops:iterator_ops",
        "//machina/python/eager:def_function",
        "//machina/python/framework:combinations",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_spec",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:variables",
        "//machina/python/ops/ragged:ragged_tensor",
        "//machina/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "default_types",
    srcs = [
        "default_types.py",
    ],
    visibility = ["//machina:internal"],
    deps = [
        ":serialization",
        ":util",
        "//machina/core/function/trace_type:default_types_proto_py",
        "//machina/python/types:trace",
        "//machina/python/util:custom_nest_protocol",
    ],
)

py_strict_test(
    name = "default_types_test",
    srcs = ["default_types_test.py"],
    deps = [
        ":default_types",
        ":serialization",
        #internal proto upb dep
        "//machina/python/platform:client_testlib",
        "//machina/python/types:trace",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "custom_nest_trace_type",
    srcs = [
        "custom_nest_trace_type.py",
    ],
    visibility = ["//machina:internal"],
    deps = [
        ":util",
        "//machina/python/types:trace",
        "//machina/python/util:custom_nest_protocol",
    ],
)

py_strict_test(
    name = "custom_nest_trace_type_test",
    srcs = ["custom_nest_trace_type_test.py"],
    deps = [
        ":custom_nest_trace_type",
        ":default_types",
        #internal proto upb dep
        "//machina/python/platform:client_testlib",
        "//machina/python/types:trace",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "serialization",
    srcs = [
        "serialization.py",
    ],
    visibility = ["//machina:internal"],
    deps = ["//machina/core/function/trace_type:serialization_proto_py"],
)

py_strict_test(
    name = "serialization_test",
    srcs = ["serialization_test.py"],
    deps = [
        ":serialization",
        #internal proto upb dep
        "//machina/core/function/trace_type:serialization_test_proto_py",
        "//machina/python/platform:client_testlib",
    ],
)

tf_proto_library(
    name = "serialization_proto",
    srcs = [
        "serialization.proto",
    ],
    visibility = ["//machina:internal"],
)

tf_proto_library(
    name = "serialization_test_proto",
    srcs = [
        "serialization_test.proto",
    ],
    protodeps = [
        ":serialization_proto",
    ],
    visibility = ["//machina:internal"],
)

tf_proto_library(
    name = "default_types_proto",
    srcs = [
        "default_types.proto",
    ],
    protodeps = [
        ":serialization_proto",
    ],
    visibility = ["//machina:internal"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "serialization_py_pb2",
#     visibility = ["//machina:internal"],
#     deps = [":serialization_proto"],
# )
#
# py_proto_library(
#     name = "serialization_test_py_pb2",
#     visibility = ["//machina:internal"],
#     deps = [":serialization_test_proto"],
# )
#
# py_proto_library(
#     name = "default_types_py_pb2",
#     visibility = ["//machina:internal"],
#     deps = [":default_types_proto"],
# )
# copybara:uncomment_end
