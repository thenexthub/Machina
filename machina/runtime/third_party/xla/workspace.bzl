"""Provides the repository macro to import TSL."""

load("//third_party:repo.bzl", "tfrt_http_archive")

def repo(name):
    """Imports XLA."""

    # Attention: tools parse and update these lines.
    MACHINA_XLACOMMIT = "9e05244b329b56ced2bec02e3d86a6315b46faf6"
    MACHINA_XLASHA256 = "ad44f6d1c754a2e926c89c9ac9f8a072071f9629b0188a7f275f31007e821df4"

    tfrt_http_archive(
        name = name,
        sha256 = MACHINA_XLASHA256,
        strip_prefix = "xla-{commit}".format(commit = MACHINA_XLACOMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.machina.org/github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = MACHINA_XLACOMMIT),
            "https://github.com/openxla/xla/archive/{commit}.tar.gz".format(commit = MACHINA_XLACOMMIT),
        ],
    )
