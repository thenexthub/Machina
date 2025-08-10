# Description: Tensorflow Serving.

package(default_visibility = ["//machina_serving:internal"])

licenses(["notice"])

exports_files(["LICENSE"])

# open source marker; do not remove

package_group(
    name = "internal",
    packages = [
        "//learning/gemini/deployment/disaggregation/...",
        "//machina_serving/...",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
)
