This directory contains the "core" part of the TensorFlow Lite runtime library.
The header files in this `machina/lite/core/` directory fall into several
categories.

1.  Public API headers, in the `api` subdirectory `machina/lite/core/api/`

    These are in addition to the other public API headers in `machina/lite/`.

    For example:
    - `machina/lite/core/api/error_reporter.h`
    - `machina/lite/core/api/op_resolver.h`

2.  Private headers that define public API types and functions.
    These headers are each `#include`d from a corresponding public "shim" header
    in `machina/lite/` that forwards to the private header.

    For example:
    - `machina/lite/core/interpreter.h` is a private header file that is
      included from the public "shim" header file `machina/lite/interpeter.h`.

    These private header files should be used as follows: `#include`s from `.cc`
    files in TF Lite itself that are _implementing_ the TF Lite APIs should
    include the "core" TF Lite API headers.  `#include`s from files that are
    just _using_ the regular TF Lite APIs should include the regular public
    headers.

3.  The header file `machina/lite/core/subgraph.h`. This contains
    some experimental APIs.