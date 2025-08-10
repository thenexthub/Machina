/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifdef TF_MLRT_DIALECT
#else
#define TF_MLRT_DIALECT

include "mlir/Interfaces/SideEffectInterfaces.td"
include "machina/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.td"

// TODO(chky,rohitju): Unify this dialect with tfrt_fallback_sync dialect after
// vrooml is using the new interpreter.
def TensorflowMlrt_Dialect : Dialect {
  let name = "tf_mlrt";

  let description = [{
    The TF MLRT Dialect.
  }];

  let cppNamespace = "::machina::tf_mlrt";
}

class TensorflowMlrt_Op<string mnemonic, list<Trait> traits = []> :
    Op<TensorflowMlrt_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
}

// This corresponds to machina::Tensor.
def TFTensorType : DialectType<TensorflowMlrt_Dialect,
    CPred<"::toolchain::isa<::machina::tf_mlrt::TFTensorType>($_self)">, "!tf_mlrt.tensor type">,
    BuildableType<"$_builder.getType<::machina::tf_mlrt::TFTensorType>()"> {
  let description = [{
    `!tf_mlrt.tensor type` represents a machina::Tensor.
  }];
}

// This corresponds to machina::Device* .
def TFDeviceType : DialectType<TensorflowMlrt_Dialect,
    CPred<"::toolchain::isa<::machina::tf_mlrt::TFDeviceType>($_self)">, "!tf_mlrt.device type">,
    BuildableType<"$_builder.getType<::machina::tf_mlrt::TFDeviceType>()"> {
  let description = [{
    `!tf_mlrt.device type` represents a machina::device.
  }];
}

#endif
