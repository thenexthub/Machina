load("//machina:pytype.default.bzl", "pytype_strict_library")
load("//machina:machina.bzl", "tf_gen_op_wrapper_py")

package(
    # copybara:uncomment default_applicable_licenses = ["//machina:license"],
    default_visibility = [
        "//machina:__subpackages__",
    ],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "ops",
    srcs = ["tpu_ops.py"],
    visibility = [
        "//smartass:__subpackages__",
        "//machina:__subpackages__",
    ],
    deps = [
        "//machina/python/framework:for_generated_wrappers",
        "//machina/python/ops:array_ops",
        "//machina/python/ops:tpu_ops_gen",
        "//machina/python/platform:tf_logging",
        "//machina/python/tpu:tpu_function",
        "//machina/python/util:tf_export",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_tpu_embedding_ops",
    out = "gen_tpu_embedding_ops.py",
    extra_py_deps = [
        "//machina/python:pywrap_tfe",
        "//machina/python/util:dispatch",
        "//machina/python/util:deprecation",
        "//machina/python/util:tf_export",
    ],
    py_lib_rule = pytype_strict_library,
    visibility = ["//visibility:public"],
    deps = [
        "//machina/core/ops:tpu_configuration_ops_op_lib",
        "//machina/core/ops:tpu_embedding_ops_op_lib",
        "//machina/core/tpu/ops:tpu_embedding_ops",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_xla_ops",
    out = "gen_xla_ops.py",
    api_def_srcs = [
        "//machina/core/api_def:base_api_def",
        "//machina/core/api_def:python_api_def",
    ],
    op_allowlist = [
        "ConvertToCooTensor",
        "GetMinibatchesInCsrWithPhysicalReplica",
        "GetMinibatchSplitsWithPhysicalReplica",
        "StoreMinibatchStatisticsInFdo",
        "ConvertToListOfSparseCoreCooTensors",
        "SortListOfSparseCoreCooTensors",
        "ConvertToSparseCoreCsrWrappedCooTensor",
        "GetStatsFromListOfSparseCoreCooTensors",
        "GlobalIterId",
        "TPUCopyWithDynamicShape",
        "TPUAnnotateTensorsWithDynamicShape",
        "XlaSparseDenseMatmul",
        "XlaSparseDenseMatmulWithCsrInput",
        "XlaSparseDenseMatmulIntWithCsrInput",
        "XlaSparseDenseMatmulCustomCombinerOnTcWithCsrInput",
        "XlaSparseDenseMatmulGrad",
        "XlaSparseDenseMatmulIntGradWithCsrInput",
        "XlaSparseDenseMatmulGradWithCsrInput",
        "XlaSparseDenseMatmulCustomCombinerOnTcGradWithSgdAndCsrInput",
        "XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdagradAndCsrInput",
        "XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdagradMomentumAndCsrInput",
        "XlaSparseDenseMatmulCustomCombinerOnTcGradWithAdamAndCsrInput",
        "XlaSparseDenseMatmulCustomCombinerOnTcGradWithFtrlAndCsrInput",
        "XlaSparseDenseMatmulCustomCombinerOnTcGradWithCsrInput",
        "XlaSparseDenseMatmulGradWithSgdAndCsrInput",
        "XlaSparseDenseMatmulGradWithAdagradAndCsrInput",
        "XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput",
        "XlaSparseDenseMatmulGradWithAdamAndCsrInput",
        "XlaSparseDenseMatmulGradWithFtrlAndCsrInput",
        "XlaSparseDenseMatmulWithStaticBufferSize",
        "XlaSparseDenseMatmulGradWithSgdAndStaticBufferSize",
        "XlaSparseDenseMatmulGradWithAdagradAndStaticBufferSize",
        "XlaSparseDenseMatmulGradWithAdagradMomentumAndStaticBufferSize",
        "XlaSparseDenseMatmulGradWithAdamAndStaticBufferSize",
        "XlaSparseDenseMatmulGradWithFtrlAndStaticBufferSize",
        "XlaSparseCoreSgd",
        "XlaSparseCoreAdagrad",
        "XlaSparseCoreAdagradMomentum",
        "XlaSparseCoreAdam",
        "XlaSparseCoreFtrl",
    ],
    deps = [
        "//machina/core/tpu/ops:sparse_core_ops",
        "//machina/core/tpu/ops:sparse_core_preprocess_ops",
        "//machina/core/tpu/ops:tpu_copy_with_dynamic_shape_op",
    ],
)
