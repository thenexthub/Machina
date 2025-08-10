load("//machina:strict.default.bzl", "py_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "generate_v2_renames_map",
    srcs = ["generate_v2_renames_map.py"],
    deps = [
        "//machina:machina_py",
        "//machina/python:modules_with_exports",
        "//machina/python:no_contrib",
        "//machina/python/lib/io:file_io",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
        "//machina/tools/common:public_api",
        "//machina/tools/common:traverse",
        "//machina/tools/compatibility:all_renames_v2",
        "@absl_py//absl:app",
    ],
)

py_strict_binary(
    name = "generate_v2_reorders_map",
    srcs = ["generate_v2_reorders_map.py"],
    deps = [
        "//machina:machina_py",
        "//machina/python:no_contrib",
        "//machina/python/lib/io:file_io",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
        "//machina/tools/common:public_api",
        "//machina/tools/common:traverse",
        "//machina/tools/compatibility:tf_upgrade_v2_lib",
        "@absl_py//absl:app",
    ],
)
