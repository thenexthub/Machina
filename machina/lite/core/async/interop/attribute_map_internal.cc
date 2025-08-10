/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Sunday, June 15, 2025.
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
#include "machina/lite/core/async/interop/attribute_map_internal.h"

#include "machina/lite/core/async/interop/reconcile_fns.h"

namespace tflite {
namespace interop {

bool AttributeMap::ReconcileAttributes(const AttributeMap* other,
                                       AttributeMap* merged,
                                       AttributeMap* conflict) const {
  if (other == nullptr || merged == nullptr) return false;
  if (type_ != other->type_) return false;
  merged->type_ = type_;
  if (conflict) conflict->type_ = type_;

  // TODO(b/191883048): Reconcile custom keys.
  return tflite::interop::ReconcileGeneralAttributeKeys(
      type_, &attrs_, &other->attrs_, &merged->attrs_,
      conflict ? &conflict->attrs_ : nullptr);
}

bool AttributeMap::CheckAttributeCoverage(const AttributeMap* other,
                                          AttributeMap* conflict) const {
  if (other == nullptr) return false;
  if (type_ != other->type_) return false;
  if (conflict) conflict->type_ = type_;

  // TODO(b/191883048): Check custom key coverage.
  return tflite::interop::CheckGeneralAttributeKeysCoverage(
      type_, &attrs_, &other->attrs_, conflict ? &conflict->attrs_ : nullptr);
}

}  // namespace interop
}  // namespace tflite
