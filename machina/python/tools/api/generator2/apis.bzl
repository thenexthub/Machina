"""generate_api API definitions."""

load(":patterns.bzl", "compile_patterns")

APIS = {
    "tf_keras": {
        "decorator": "machina.python.util.tf_export.keras_export",
        "target_patterns": compile_patterns([
            "//third_party/py/tf_keras/...",
        ]),
    },
    "machina": {
        "decorator": "machina.python.util.tf_export.tf_export",
        "target_patterns": compile_patterns([
            "//machina/python/...",
            "//machina/dtensor/python:accelerator_util",
            "//machina/dtensor/python:api",
            "//machina/dtensor/python:config",
            "//machina/dtensor/python:d_checkpoint",
            "//machina/dtensor/python:d_variable",
            "//machina/dtensor/python:input_util",
            "//machina/dtensor/python:layout",
            "//machina/dtensor/python:mesh_util",
            "//machina/dtensor/python:tpu_util",
            "//machina/dtensor/python:save_restore",
            "//machina/lite/python/...",
            "//machina/python:modules_with_exports",
            "//machina/lite/tools/optimize/debugging/python:all",
            "//machina/compiler/mlir/quantization/machina/python:all",
        ]),
    },
}
