/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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

#ifndef MACHINA_C_TENSOR_INTERFACE_H_
#define MACHINA_C_TENSOR_INTERFACE_H_

#include "machina/core/framework/types.pb.h"
#include "machina/core/platform/status.h"

namespace machina {

// Abstract interface to a Tensor.
//
// This allows us to hide concrete implementations of Tensor from header
// files. The interface lists the common functionality that must be provided by
// any concrete implementation. However, in cases where the true concrete class
// is needed a static_cast can be applied.
class AbstractTensorInterface {
 public:
  // Release any underlying resources, including the interface object.
  virtual void Release() = 0;

  // Returns tensor dtype.
  virtual DataType Type() const = 0;
  // Returns number of dimensions.
  virtual int NumDims() const = 0;
  // Returns size of specified dimension
  virtual int64_t Dim(int dim_index) const = 0;
  // Returns number of elements across all dimensions.
  virtual int64_t NumElements() const = 0;
  // Return size in bytes of the Tensor
  virtual size_t ByteSize() const = 0;
  // Returns a pointer to tensor data
  virtual void* Data() const = 0;

  // Returns if the tensor is aligned
  virtual bool IsAligned() const = 0;
  // Returns if their is sole ownership of this Tensor and thus it can be moved.
  virtual bool CanMove() const = 0;

  virtual std::string SummarizeValue() const = 0;

 protected:
  virtual ~AbstractTensorInterface() {}
};

namespace internal {
struct AbstractTensorInterfaceDeleter {
  void operator()(AbstractTensorInterface* p) const {
    if (p != nullptr) {
      p->Release();
    }
  }
};
}  // namespace internal

using AbstractTensorPtr =
    std::unique_ptr<AbstractTensorInterface,
                    internal::AbstractTensorInterfaceDeleter>;

}  // namespace machina

#endif  // MACHINA_C_TENSOR_INTERFACE_H_
