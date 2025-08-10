load("//machina:strict.default.bzl", "py_strict_test")
load("//machina:machina.default.bzl", "pybind_extension")

pybind_extension(
    name = "register_list_ops_py",
    srcs = ["register_list_ops_py.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "register_list_ops_py.pyi",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/lite:mutable_op_resolver",
        "//machina/lite/kernels/variants:list_ops_lib",
        "@pybind11",
    ],
)

py_strict_test(
    name = "end_to_end_test",
    srcs = ["end_to_end_test.py"],
    tags = ["nochromiumos_arm"],
    deps = [
        ":register_list_ops_py",
        "@absl_py//absl/testing:parameterized",
        #internal proto upb dep
        "//third_party/py/numpy",
        "//machina:machina_py",
        "//machina/lite/python:interpreter",
        "//machina/lite/python:lite",
        "//machina/python/ops:list_ops",
        "//machina/python/platform:test",
    ],
)
