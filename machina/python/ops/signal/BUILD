load("//machina:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "dct_ops",
    srcs = ["dct_ops.py"],
    deps = [
        ":fft_ops",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:smart_cond",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

py_strict_library(
    name = "fft_ops",
    srcs = ["fft_ops.py"],
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:manip_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:spectral_ops_gen",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "mel_ops",
    srcs = ["mel_ops.py"],
    deps = [
        ":shape_ops",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "mfcc_ops",
    srcs = ["mfcc_ops.py"],
    deps = [
        ":dct_ops",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "reconstruction_ops",
    srcs = ["reconstruction_ops.py"],
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
    ],
)

py_strict_library(
    name = "shape_ops",
    srcs = ["shape_ops.py"],
    deps = [
        ":util_ops",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "signal",
    srcs = ["signal.py"],
    deps = [
        ":__init__",
        ":dct_ops",
        ":fft_ops",
        ":mel_ops",
        ":mfcc_ops",
        ":reconstruction_ops",
        ":shape_ops",
        ":spectral_ops",
        ":window_ops",
    ],
)

py_strict_library(
    name = "spectral_ops",
    srcs = ["spectral_ops.py"],
    deps = [
        ":dct_ops",
        ":fft_ops",
        ":reconstruction_ops",
        ":shape_ops",
        ":window_ops",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "util_ops",
    srcs = ["util_ops.py"],
    deps = [
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:while_loop",
    ],
)

py_strict_library(
    name = "window_ops",
    srcs = ["window_ops.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_util",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:special_math_ops",
        "//machina/python/util:dispatch",
        "//machina/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)
