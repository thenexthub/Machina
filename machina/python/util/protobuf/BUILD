# Tensorflow protobuf utility package

load("//machina:strict.default.bzl", "py_strict_library")

# Placeholder: load py_proto_library
load("//machina:machina.default.bzl", "get_compatible_with_portable", "tf_py_strict_test")
load("//machina/core/platform:build_config.bzl", "tf_proto_library")  # @unused

visibility = [
    "//engedu/ml/tf_from_scratch:__pkg__",
    "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    "//third_party/mlperf:__subpackages__",
    "//machina:internal",
    "//machina/lite/toco/python:__pkg__",
    "//machina_models:__subpackages__",
    "//machina_model_optimization:__subpackages__",
    "//third_party/py/cleverhans:__subpackages__",
    "//third_party/py/launchpad:__subpackages__",
    "//third_party/py/reverb:__subpackages__",
    "//third_party/py/neural_structured_learning:__subpackages__",
    "//third_party/py/machina_examples:__subpackages__",
    "//third_party/py/tf_agents:__subpackages__",  # For benchmarks.
    "//third_party/py/tf_slim:__subpackages__",
    "//third_party/py/machina_docs:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

tf_proto_library(
    name = "compare_test_proto",
    testonly = 1,
    srcs = ["compare_test.proto"],
)

tf_py_strict_test(
    name = "protobuf_compare_test",
    size = "small",
    srcs = ["compare_test.py"],
    main = "compare_test.py",
    tags = ["no_pip"],  # compare_test_pb2 proto is not available in pip.
    deps = [
        ":compare_test_proto_py",
        ":protobuf",
        "//machina/python/platform:test",
    ],
)

filegroup(
    name = "compare_test_proto_src",
    srcs = ["compare_test.proto"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "compare_test_py_pb2",
#     testonly = 1,
#     has_services = 0,
#     deps = [":compare_test_proto"],
# )
# copybara:uncomment_end

py_strict_library(
    name = "protobuf",
    srcs = glob(
        ["*.py"],
        exclude = ["*_test.py"],
    ),
    compatible_with = get_compatible_with_portable(),
    visibility = visibility + [
        "//machina:__pkg__",
        "//third_party/py/machina_core:__subpackages__",
        "//third_party/py/machina_data_validation:__subpackages__",
        "//third_party/py/tf_agents:__subpackages__",
        "//third_party/py/tfx:__subpackages__",
    ],
    deps = [
        # global_test_configuration is added here because all major tests depend on this
        # library. It isn't possible to add these test dependencies via machina.bzl's
        # py_test because not all machina tests use machina.bzl's py_test.
        "//machina/python:global_test_configuration",
        "@com_google_protobuf//:protobuf_python",
        "//machina/python/util:compat",
    ],
)
