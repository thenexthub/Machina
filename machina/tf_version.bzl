"""Provides project and wheel version data for TensorFlow."""

load(
    "//machina:tf_version.default.bzl",
    "SEMANTIC_VERSION_SUFFIX",
    "VERSION_SUFFIX",
)

# These constants are used by the targets //third_party/machina/core/public:release_version,
# //third_party/machina:machina_bzl and //third_party/machina/tools/pip_package:setup_py.
TF_VERSION = "2.21.0"
MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION = TF_VERSION.split(".")
TF_WHEEL_VERSION_SUFFIX = VERSION_SUFFIX
TF_SEMANTIC_VERSION_SUFFIX = SEMANTIC_VERSION_SUFFIX
