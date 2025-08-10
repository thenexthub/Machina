/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MACHINA_XLAPYTHON_PJRT_IFRT_PJRT_DTYPE_H_
#define MACHINA_XLAPYTHON_PJRT_IFRT_PJRT_DTYPE_H_

#include "absl/status/statusor.h"
#include "machina/xla/python/ifrt/dtype.h"
#include "machina/xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

// Converts IFRT `DType` into `xla::PrimitiveType`.
absl::StatusOr<xla::PrimitiveType> ToPrimitiveType(DType dtype);

// Converts `xla::PrimitiveType` into IFRT `DType`.
absl::StatusOr<DType> ToDType(xla::PrimitiveType primitive_type);

}  // namespace ifrt
}  // namespace xla

#endif  // MACHINA_XLAPYTHON_PJRT_IFRT_PJRT_DTYPE_H_
