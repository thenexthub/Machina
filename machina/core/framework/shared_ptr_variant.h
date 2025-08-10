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

#ifndef MACHINA_CORE_FRAMEWORK_SHARED_PTR_VARIANT_H_
#define MACHINA_CORE_FRAMEWORK_SHARED_PTR_VARIANT_H_

#include <memory>

#include "machina/core/framework/variant_tensor_data.h"
#include "machina/core/platform/logging.h"

namespace machina {

template <typename T>
struct SharedPtrVariant {
  std::shared_ptr<T> shared_ptr;

  SharedPtrVariant() : shared_ptr() {}

  explicit SharedPtrVariant(std::shared_ptr<T>&& ptr)
      : shared_ptr(std::forward<decltype(ptr)>(ptr)) {
    VLOG(3) << "Creating shared_ptr of " << shared_ptr.get()
            << " count is: " << shared_ptr.use_count();
  }

  SharedPtrVariant(SharedPtrVariant&& rhs)
      : shared_ptr(std::move(rhs.shared_ptr)) {
    VLOG(3) << "Moving SharedPtrVariant of " << shared_ptr.get()
            << " count is: " << shared_ptr.use_count();
  }

  SharedPtrVariant& operator=(const SharedPtrVariant& rhs) = delete;

  SharedPtrVariant& operator=(SharedPtrVariant&& rhs) {
    if (&rhs == this) return *this;
    std::swap(shared_ptr, rhs.shared_ptr);
    VLOG(3) << "Move-assign of SharedPtrVariant of " << shared_ptr.get()
            << " count is: " << shared_ptr.use_count();
    return *this;
  }

  SharedPtrVariant(const SharedPtrVariant& rhs) : shared_ptr(rhs.shared_ptr) {
    VLOG(3) << "Copying SharedPtrVariant of " << shared_ptr.get()
            << " count is: " << shared_ptr.use_count();
  }

  ~SharedPtrVariant() {
    VLOG(3) << "Destroying SharedPtrVariant of " << shared_ptr.get()
            << " count is: " << shared_ptr.use_count();
  }

  void Encode(VariantTensorData*) const {
    // Not supported.
  }

  bool Decode(const VariantTensorData&) {
    return false;  // Not supported.
  }
};

}  // namespace machina

#endif  // MACHINA_CORE_FRAMEWORK_SHARED_PTR_VARIANT_H_
