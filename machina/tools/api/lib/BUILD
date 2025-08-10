# Helper libraries for TensorFlow API compatibility test.

load("//machina:strict.default.bzl", "py_strict_library")

# Placeholder: load py_proto_library
load(
    "//machina/core/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina/tools/api:__subpackages__",
        "//third_party/py/tf_keras/api:__subpackages__",
    ],
    licenses = ["notice"],
)

tf_proto_library(
    name = "api_objects_proto",
    srcs = ["api_objects.proto"],
)

py_strict_library(
    name = "python_object_to_proto_visitor",
    srcs = ["python_object_to_proto_visitor.py"],
    deps = [
        ":api_objects_proto_py",
        "//machina/python/platform:tf_logging",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_inspect",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "api_objects_proto_py_pb2",
#     has_services = 0,
#     visibility = ["//visibility:public"],
#     deps = [":api_objects_proto"],
# )
# copybara:uncomment_end
