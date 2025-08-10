/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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

#ifndef MACHINA_CORE_FRAMEWORK_TYPED_ALLOCATOR_H_
#define MACHINA_CORE_FRAMEWORK_TYPED_ALLOCATOR_H_

#include <limits>

#include "machina/core/framework/allocator.h"
#include "machina/core/framework/resource_handle.h"
#include "machina/core/framework/type_traits.h"
#include "machina/core/platform/types.h"

namespace machina {

class Variant;

// Convenience functions to do typed allocation.  C++ constructors
// and destructors are invoked for complex types if necessary.
class TypedAllocator {
 public:
  // May return NULL if the tensor has too many elements to represent in a
  // single allocation.
  template <typename T>
  static T* Allocate(Allocator* raw_allocator, size_t num_elements,
                     const AllocationAttributes& allocation_attr) {
    // TODO(jeff): Do we need to allow clients to pass in alignment
    // requirements?

    if (num_elements > (std::numeric_limits<size_t>::max() / sizeof(T))) {
      return nullptr;
    }

    void* p =
        raw_allocator->AllocateRaw(Allocator::kAllocatorAlignment,
                                   sizeof(T) * num_elements, allocation_attr);
    T* typed_p = reinterpret_cast<T*>(p);
    if (typed_p) RunCtor<T>(raw_allocator, typed_p, num_elements);
    return typed_p;
  }

  template <typename T>
  static void Deallocate(Allocator* raw_allocator, T* ptr,
                         size_t num_elements) {
    if (ptr) {
      RunDtor<T>(raw_allocator, ptr, num_elements);
      raw_allocator->DeallocateRaw(ptr, Allocator::kAllocatorAlignment,
                                   sizeof(T) * num_elements);
    }
  }

 private:
  // No constructors or destructors are run for simple types
  template <typename T>
  static void RunCtor(Allocator* raw_allocator, T* p, size_t n) {
    static_assert(is_simple_type<T>::value, "T is not a simple type.");
  }

  template <typename T>
  static void RunDtor(Allocator* raw_allocator, T* p, size_t n) {}

  static void RunVariantCtor(Variant* p, size_t n);

  static void RunVariantDtor(Variant* p, size_t n);
};

template <>
/* static */
inline void TypedAllocator::RunCtor(Allocator* raw_allocator, tstring* p,
                                    size_t n) {
  if (!raw_allocator->AllocatesOpaqueHandle()) {
    for (size_t i = 0; i < n; ++p, ++i) new (p) tstring();
  }
}

template <>
/* static */
inline void TypedAllocator::RunDtor(Allocator* raw_allocator, tstring* p,
                                    size_t n) {
  if (!raw_allocator->AllocatesOpaqueHandle()) {
    for (size_t i = 0; i < n; ++p, ++i) p->~tstring();
  }
}

template <>
/* static */
inline void TypedAllocator::RunCtor(Allocator* raw_allocator, ResourceHandle* p,
                                    size_t n) {
  if (!raw_allocator->AllocatesOpaqueHandle()) {
    for (size_t i = 0; i < n; ++p, ++i) new (p) ResourceHandle();
  }
}

template <>
/* static */
inline void TypedAllocator::RunDtor(Allocator* raw_allocator, ResourceHandle* p,
                                    size_t n) {
  if (!raw_allocator->AllocatesOpaqueHandle()) {
    for (size_t i = 0; i < n; ++p, ++i) p->~ResourceHandle();
  }
}

template <>
/* static */
inline void TypedAllocator::RunCtor(Allocator* raw_allocator, Variant* p,
                                    size_t n) {
  if (!raw_allocator->AllocatesOpaqueHandle()) {
    RunVariantCtor(p, n);
  }
}

template <>
/* static */
inline void TypedAllocator::RunDtor(Allocator* raw_allocator, Variant* p,
                                    size_t n) {
  if (!raw_allocator->AllocatesOpaqueHandle()) {
    RunVariantDtor(p, n);
  }
}

}  // namespace machina

#endif  // MACHINA_CORE_FRAMEWORK_TYPED_ALLOCATOR_H_
