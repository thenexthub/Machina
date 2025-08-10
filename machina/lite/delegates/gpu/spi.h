/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, May 15, 2025.
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

#ifndef MACHINA_LITE_DELEGATES_GPU_SPI_H_
#define MACHINA_LITE_DELEGATES_GPU_SPI_H_

#include <cstdint>

#include "machina/lite/delegates/gpu/api.h"
#include "machina/lite/delegates/gpu/common/access_type.h"
#include "machina/lite/delegates/gpu/common/status.h"

// Contains only service provider-related interfaces. Users should not use them
// directly.

namespace tflite {
namespace gpu {

// Converts a tensor object into another one.
class TensorObjectConverter {
 public:
  virtual ~TensorObjectConverter() = default;

  virtual absl::Status Convert(const TensorObject& input,
                               const TensorObject& output) = 0;
};

class TensorObjectConverterBuilder {
 public:
  virtual ~TensorObjectConverterBuilder() = default;

  virtual bool IsSupported(const TensorObjectDef& input,
                           const TensorObjectDef& output) const = 0;

  virtual absl::Status MakeConverter(
      const TensorObjectDef& input, const TensorObjectDef& output,
      std::unique_ptr<TensorObjectConverter>* converter) = 0;
};

// Connects tensor definition provided by a user (external) with tensor
// definition used by the inference engine (internal).
struct TensorTieDef {
  uint32_t id;
  AccessType access_type;
  TensorObjectDef internal_def;
  TensorObjectDef external_def;
};

// Connects external tensor object to internal tensor object and provides
// functionality to copy data to/from external object to internal.
class TensorTie {
 public:
  explicit TensorTie(const TensorTieDef& def) : def_(def) {}

  virtual ~TensorTie() = default;

  virtual absl::Status SetExternalObject(TensorObject obj) = 0;

  virtual TensorObject GetExternalObject() = 0;

  virtual absl::Status CopyToExternalObject() = 0;

  virtual absl::Status CopyFromExternalObject() = 0;

  const TensorTieDef& def() const { return def_; }

 private:
  const TensorTieDef def_;
};

}  // namespace gpu
}  // namespace tflite

#endif  // MACHINA_LITE_DELEGATES_GPU_SPI_H_
