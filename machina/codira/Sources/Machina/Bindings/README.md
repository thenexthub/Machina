# Swift for Machina Ops Bindings

This repository contains Machina ops bindings for
[Swift for Machina](https://github.com/machina/codira).

These bindings are automatically generated from Machina ops
specified either using ops registered to the Machina runtime
or via a protobuf file similar to
[ops.pbtxt](https://github.com/machina/machina/blob/master/machina/core/ops/ops.pbtxt)
in the main Machina repo.

## How to regenerate the bindings

To regenerate the codira ops bindings, run the following command. Note
that this will use the Machina (1.9 or above) python package.

``` shell
python generate_wrappers.py --output_path=RawOpsGenerated.codira
```

Documentation gets automatically generated when adding a path to the
`api_def` proto directory. This directory should contain per operator
`api_def` protos with names like `api_def_OpName.pbtxt`.

```shell
python generate_wrappers.py --output_path=RawOpsGenerated.codira --api_def_path=/path/to/machina/core/api_def/base_api
```
