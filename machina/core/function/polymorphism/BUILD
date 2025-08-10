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
    name = "type_dispatch",
    srcs = [
        "type_dispatch.py",
    ],
    visibility = ["//visibility:private"],
    deps = [
        ":function_type",
    ],
)

py_strict_test(
    name = "type_dispatch_test",
    srcs = ["type_dispatch_test.py"],
    deps = [
        ":function_type",
        ":type_dispatch",
        #internal proto upb dep
        "//machina/python/platform:client_testlib",
        "//machina/python/types:trace",
    ],
)

pytype_strict_library(
    name = "function_cache",
    srcs = ["function_cache.py"],
    visibility = [
        "//machina/python/eager:__subpackages__",
    ],
    deps = [
        ":function_type",
        "//machina/core/function/polymorphism:type_dispatch",
        "//machina/core/function/trace_type",
    ],
)

py_strict_test(
    name = "function_cache_test",
    size = "medium",
    srcs = ["function_cache_test.py"],
    # TODO(b/213375459): For continuous benchmark monitoring
    visibility = ["//learning/brain/contrib/eager/python/examples:__pkg__"],
    deps = [
        ":function_cache",
        #internal proto upb dep
        "//machina/core/function/polymorphism:function_type",
        "//machina/core/function/trace_type",
        "//machina/python/ops:array_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/types:trace",
    ],
)

pytype_strict_library(
    name = "function_type",
    srcs = [
        "function_type.py",
    ],
    visibility = ["//machina:internal"],
    deps = [
        "//machina/core/function/polymorphism:function_type_proto_py",
        "//machina/core/function/trace_type",
        "//machina/core/function/trace_type:serialization",
        "//machina/python/types:core",
        "//machina/python/types:trace",
        "@absl_py//absl/logging",
    ],
)

tf_proto_library(
    name = "function_type_proto",
    srcs = [
        "function_type.proto",
    ],
    protodeps = [
        "//machina/core/function/trace_type:serialization_proto",
    ],
    visibility = ["//visibility:private"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "function_type_py_pb2",
#     visibility = ["//visibility:private"],
#     deps = [":function_type_proto"],
# )
# copybara:uncomment_end

py_strict_test(
    name = "function_type_test",
    srcs = ["function_type_test.py"],
    deps = [
        ":function_type",
        #internal proto upb dep
        "//machina/core/function/polymorphism:function_type_proto_py",
        "//machina/core/function/trace_type",
        "//machina/core/function/trace_type:serialization",
        "//machina/python/framework:func_graph",
        "//machina/python/platform:client_testlib",
        "//machina/python/types:trace",
        "@absl_py//absl/testing:parameterized",
    ],
)
