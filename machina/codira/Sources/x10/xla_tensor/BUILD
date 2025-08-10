licenses(["notice"])  # Apache 2.0

load(
    "//machina:machina.bzl",
    "tf_cc_binary",
    "tf_cc_shared_object",
)

cc_library(
    name = "tensor",
    srcs = glob(
        [
            "*.cpp",
            "ops/*.cpp",
        ],
        exclude = ["test.cpp"],
    ),
    hdrs = glob([
        "*.h",
        "ops/*.h",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//machina/c:c_api",
        "//machina/c:c_api_experimental",
        "//machina/c/eager:c_api",
        "//machina/c/eager:c_api_experimental",
        "//machina/compiler/tf2xla:common",
        "//machina/compiler/tf2xla/kernels:conv_op_helpers",
        "//machina/compiler/tf2xla/lib:random",
        "//machina/compiler/tf2xla/lib:scatter",
        "//machina/compiler/xla:literal",
        "//machina/compiler/xla:literal_util",
        "//machina/compiler/xla:shape_util",
        "//machina/compiler/xla:status",
        "//machina/compiler/xla:types",
        "//machina/compiler/xla:util",
        "//machina/compiler/xla:xla_data_proto_cc",
        "//machina/compiler/xla/client:padding",
        "//machina/compiler/xla/client:xla_builder",
        "//machina/compiler/xla/client/lib:arithmetic",
        "//machina/compiler/xla/client/lib:comparators",
        "//machina/compiler/xla/client/lib:constants",
        "//machina/compiler/xla/client/lib:logdet",
        "//machina/compiler/xla/client/lib:math",
        "//machina/compiler/xla/client/lib:matrix",
        "//machina/compiler/xla/client/lib:pooling",
        "//machina/compiler/xla/client/lib:prng",
        "//machina/compiler/xla/client/lib:qr",
        "//machina/compiler/xla/client/lib:self_adjoint_eig",
        "//machina/compiler/xla/client/lib:slicing",
        "//machina/compiler/xla/client/lib:svd",
        "//machina/compiler/xla/service:hlo",
        "//machina/compiler/xla/xla_client:xrt_computation_client",
        "//machina/core:core_cpu_lib",
        "//machina/core:framework",
        "//machina/core:lib",
        "//machina/core/kernels:conv_ops",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/debugging:stacktrace",
        "@com_google_absl//absl/debugging:symbolize",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

filegroup(
    name = "get_x10_dll_import_lib",
    srcs = [":x10.dll"],
    output_group = "interface_library",
    visibility = ["//visibility:public"],
)

genrule(
    name = "x10_dll_import_lib",
    srcs = [":get_x10_dll_import_lib"],
    outs = ["x10.lib"],
    cmd = select({
        "//machina:windows": "cp -f $< $@",
        "//conditions:default": "touch $@",  # Just a placeholder for Unix platforms
    }),
    visibility = ["//visibility:public"],
)

tf_cc_shared_object(
    name = "x10",
    linkopts = select({
        "//machina:macos": [
            # TODO
        ],
        "//machina:windows": [
        ],
        "//conditions:default": [
            "-z defs",
            "-s",
            "-Wl,--version-script,$(location :tf_version_script.lds)",
        ],
    }),
    per_os_targets = True,
    visibility = ["//visibility:public"],
    deps = [
        "//codira_bindings:device_wrapper",
        "//codira_bindings:xla_tensor_wrapper",
        "//codira_bindings:xla_tensor_tf_ops",
        ":tensor",
        ":tf_exported_symbols.lds",
        ":tf_version_script.lds",
    ],
)
