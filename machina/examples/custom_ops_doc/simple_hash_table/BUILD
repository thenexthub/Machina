# Build simple_hash_table custom ops example, which is similar to,
# but simpler than, tf.lookup.experimental.MutableHashTable

load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.bzl", "tf_custom_op_library")
load("//machina:machina.default.bzl", "tf_py_test")

licenses(["notice"])

tf_custom_op_library(
    name = "simple_hash_table_kernel.so",
    srcs = [
        "simple_hash_table_kernel.cc",
        "simple_hash_table_op.cc",
    ],
    deps = [
        "//machina/core/lib/gtl:map_util",
        "//machina/core/platform:strcat",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

py_strict_library(
    name = "simple_hash_table_op",
    srcs = ["simple_hash_table_op.py"],
    data = ["simple_hash_table_kernel.so"],
    srcs_version = "PY3",
    deps = [
        "//machina:machina_py",
    ],
)

py_strict_library(
    name = "simple_hash_table",
    srcs = ["simple_hash_table.py"],
    srcs_version = "PY3",
    deps = [
        ":simple_hash_table_op",
        "//machina:machina_py",
    ],
)

tf_py_test(
    name = "simple_hash_table_test",
    size = "medium",  # This test blocks because it writes and reads a file,
    timeout = "short",  # but it still runs quickly.
    srcs = ["simple_hash_table_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
    ],
    deps = [
        ":simple_hash_table",
        "//machina:machina_py",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)
