load("//machina:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//machina:machina.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = ["//machina:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "feature_column_py",
    srcs = ["feature_column_lib.py"],
    deps = [
        ":feature_column",
        ":feature_column_v2",
        ":sequence_feature_column",
        ":serialization",
    ],
)

py_strict_library(
    name = "feature_column",
    srcs = ["feature_column.py"],
    deps = [
        ":utils",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/layers:layers_base",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:embedding_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:nn_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:resource_variable_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:template",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/training:checkpoint_utils",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "//machina/tools/docs:doc_controls",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "feature_column_v2",
    srcs = [
        "feature_column_v2.py",
    ],
    deps = [
        ":feature_column",
        ":feature_column_v2_types",
        ":serialization",
        ":utils",
        "//machina/python/data/experimental/ops:lookup_ops",
        "//machina/python/data/ops:readers",
        "//machina/python/eager:context",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:tensor",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:array_ops_stack",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:cond",
        "//machina/python/ops:embedding_ops",
        "//machina/python/ops:init_ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:string_ops",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:gfile",
        "//machina/python/platform:tf_logging",
        "//machina/python/trackable:autotrackable",
        "//machina/python/trackable:base",
        "//machina/python/trackable:data_structures",
        "//machina/python/training:checkpoint_utils",
        "//machina/python/util:compat",
        "//machina/python/util:deprecation",
        "//machina/python/util:nest",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
        "//machina/tools/docs:doc_controls",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "serialization",
    srcs = [
        "serialization.py",
    ],
    deps = [
        ":feature_column_v2_types",
        "//machina/python/ops:init_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_decorator_py",
        "//machina/python/util:tf_export",
        "//machina/python/util:tf_inspect",
        "//machina/tools/docs:doc_controls",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "sequence_feature_column",
    srcs = ["sequence_feature_column.py"],
    deps = [
        ":feature_column_v2",
        ":serialization",
        ":utils",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:tensor_shape",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:check_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
        "//machina/tools/docs:doc_controls",
    ],
)

filegroup(
    name = "vocabulary_testdata",
    srcs = [
        "testdata/embedding.ckpt.data-00000-of-00001",
        "testdata/embedding.ckpt.index",
        "testdata/embedding.ckpt.meta",
        "testdata/unicode_vocabulary",
        "testdata/unicode_vocabulary.tfrecord.gz",
        "testdata/warriors_vocabulary.tfrecord.gz",
        "testdata/warriors_vocabulary.txt",
        "testdata/wire_vocabulary.tfrecord.gz",
        "testdata/wire_vocabulary.txt",
    ],
)

tf_py_strict_test(
    name = "feature_column_test",
    srcs = ["feature_column_test.py"],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_oss",  # TODO(b/206860622): Broken with numpy 1.20+
        "no_pip",
        "no_rocm",
        "no_windows",
    ],
    deps = [
        ":feature_column_test_main_lib",
    ],
)

py_strict_library(
    name = "feature_column_test_main_lib",
    srcs = ["feature_column_test.py"],
    data = [":vocabulary_testdata"],
    deps = [
        ":feature_column",
        ":feature_column_v2",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "feature_column_v2_test",
    srcs = ["feature_column_v2_test.py"],
    shard_count = 5,
    tags = [
        "no_cuda_on_cpu_tap",
        "no_oss",  # TODO(b/206860622): Broken with numpy 1.20+
        "no_pip",
        "no_rocm",
        "no_windows",
    ],
    deps = [":feature_column_v2_test_main_lib"],
)

py_strict_library(
    name = "feature_column_v2_test_main_lib",
    srcs = ["feature_column_v2_test.py"],
    data = [":vocabulary_testdata"],
    deps = [
        ":feature_column",
        ":feature_column_v2",
        ":feature_column_v2_types",
        ":serialization",
        "//machina/core:protos_all_py",
        "//machina/python/client:session",
        "//machina/python/eager:backprop",
        "//machina/python/eager:context",
        "//machina/python/framework:constant_op",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:errors",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:parsing_ops",
        "//machina/python/ops:partitioned_variables",
        "//machina/python/ops:variable_scope",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    deps = [
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/util:nest",
        "@six_archive//:six",
    ],
)

tf_py_strict_test(
    name = "sequence_feature_column_test",
    srcs = ["sequence_feature_column_test.py"],
    tags = ["no_oss"],  # Due to the usage of keras component.
    deps = [
        ":feature_column_v2",
        ":sequence_feature_column",
        ":serialization",
        "//machina/python/client:session",
        "//machina/python/framework:dtypes",
        "//machina/python/framework:ops",
        "//machina/python/framework:sparse_tensor",
        "//machina/python/framework:test_lib",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:lookup_ops",
        "//machina/python/ops:math_ops",
        "//machina/python/ops:sparse_ops",
        "//machina/python/ops:variables",
        "//machina/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "sequence_feature_column_integration_test",
    srcs = ["sequence_feature_column_integration_test.py"],
    tags = ["no_pip"],
    deps = [
        ":feature_column_v2",
        ":sequence_feature_column",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//machina/core:protos_all_py",
        "//machina/python/ops:parsing_ops",
        "//machina/python/platform:client_testlib",
        "//machina/python/util:compat",
    ],
)

tf_py_strict_test(
    name = "serialization_test",
    srcs = ["serialization_test.py"],
    deps = [
        ":feature_column_v2",
        ":serialization",
        "//machina/python/platform:client_testlib",
    ],
)

py_strict_library(
    name = "feature_column_v2_types",
    srcs = ["feature_column_v2_types.py"],
    deps = ["//machina/python/util:tf_export"],
)
