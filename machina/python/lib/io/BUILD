# python/lib/io package

load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.bzl", "if_oss")
load("//machina:machina.default.bzl", "tf_py_strict_test", "tf_python_pybind_extension")

# copybara:uncomment_begin(google-only)
# load("//third_party/zlib:BUILD_defs.bzl", "brittle_test_relying_on_stable_zlib_output")
# copybara:uncomment_end

visibility = [
    "//machina:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

tf_python_pybind_extension(
    name = "_pywrap_file_io",
    srcs = ["file_io_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_file_io.pyi",
    ],
    deps = [
        "//machina/core:framework_headers_lib",
        "//machina/core:lib",
        "//machina/core:lib_proto_parsing",
        "//machina/core:portable_gif_internal",
        "//machina/core:portable_jpeg_internal",
        "//machina/core:protos_all_cc",
        "//machina/core/platform:file_statistics",
        "//machina/python/lib/core:pybind11_absl",
        "//machina/python/lib/core:pybind11_status",
        "@com_google_absl//absl/status",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_record_io",
    srcs = ["record_io_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_record_io.pyi",
    ],
    deps = [
        "//machina/core:core_stringpiece",
        "//machina/core:lib",
        "//machina/core:lib_headers_for_pybind",
        "//machina/core:lib_proto_parsing",
        "//machina/core/platform:errors",
        "//machina/core/platform:types",
        "//machina/python/lib/core:pybind11_absl",
        "//machina/python/lib/core:pybind11_status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@pybind11",
    ],
)

py_strict_library(
    name = "file_io",
    srcs = ["file_io.py"],
    # copybara:uncomment_begin(google-only)
    # visibility = [
    # "//third_party/cloud_tpu/convergence_tools:__subpackages__",
    # "//third_party/py/tf_slim:__subpackages__",
    # "//machina:__subpackages__",
    # "//machina:internal",
    # ],
    # copybara:uncomment_end_and_comment_begin
    visibility = [
        "//visibility:public",
    ],
    # copybara:comment_end
    deps = [
        ":_pywrap_file_io",
        "//machina/python/framework:errors",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "tf_record",
    srcs = ["tf_record.py"],
    visibility = [
        "//machina:__subpackages__",
        "//machina:internal",
        "//third_party/py/tf_agents:__subpackages__",
    ],
    deps = [
        ":_pywrap_record_io",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "python_io",
    srcs = ["python_io.py"],
    visibility = [
        "//machina:__subpackages__",
        "//machina:internal",
    ],
    deps = [":tf_record"],
)

tf_py_strict_test(
    name = "file_io_test",
    size = "small",
    srcs = ["file_io_test.py"],
    tags = [
        "no_rocm",
        "no_windows",
    ],
    deps = [
        ":file_io",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
        "//machina/python/platform:gfile",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = if_oss("tf_record_test", "_tf_record_test"),
    size = "small",
    srcs = ["tf_record_test.py"],
    main = "tf_record_test.py",
    tags = [
        # copybara:uncomment_begin(google-only)
        # "manual",
        # "notap",
        # copybara:uncomment_end
    ],
    deps = [
        ":tf_record",
        "//machina/python/framework:errors",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
    ],
)

# _tf_record_test relies on stable zlib output only during its execution.
# It should not be brittle with respect to upgrades of zlib software or
# different computers using different zlib software.
# copybara:uncomment_begin(google-only)
# brittle_test_relying_on_stable_zlib_output(
#     name = "tf_record_test",
#     test_target = ":_tf_record_test",
# )
# copybara:uncomment_end
