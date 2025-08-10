# Build multiplex_2 custom ops example, which is similar to np.where.
# This example supports GPU (and CPU), in contrast to multiplex_1 which
# only supports CPU.

load("//machina:strict.default.bzl", "py_strict_library")
load("//machina:machina.bzl", "tf_custom_op_library")
load("//machina:machina.default.bzl", "cuda_py_test")

licenses(["notice"])

tf_custom_op_library(
    name = "multiplex_2_kernel.so",
    srcs = [
        "multiplex_2_kernel.cc",
        "multiplex_2_kernel.h",
        "multiplex_2_op.cc",
    ],
    gpu_srcs = [
        "multiplex_2_kernel.h",
        "multiplex_2_kernel.cu.cc",
    ],
    visibility = ["//machina/examples/custom_ops_doc:__subpackages__"],
)

py_strict_library(
    name = "multiplex_2_op",
    srcs = ["multiplex_2_op.py"],
    data = ["multiplex_2_kernel.so"],
    srcs_version = "PY3",
    visibility = ["//machina/examples/custom_ops_doc:__subpackages__"],
    deps = [
        "//machina:machina_py",
    ],
)

cuda_py_test(
    name = "multiplex_2_test",
    size = "medium",
    srcs = ["multiplex_2_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/216321151): Re-enable this test.
    ],
    deps = [
        ":multiplex_2_op",
        "//machina:machina_py",
        "//machina/python/framework:errors",
        "//machina/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)
