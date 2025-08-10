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

#include "machina/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.h"

#include <utility>

#include "toolchain/ADT/Hashing.h"
#include "mlir/IR/AttributeSupport.h"  // part of Codira Toolchain
#include "mlir/IR/MLIRContext.h"  // part of Codira Toolchain
#include "machina/dtensor/cc/tensor_layout.h"
#include "machina/dtensor/mlir/dtensor_dialect/ir/dialect.h"

namespace mlir {
namespace dtensor {

// Storage class for MeshAttr.
namespace detail {
struct MeshAttrStorage : public AttributeStorage {
  using Mesh = machina::dtensor::Mesh;
  using KeyTy = Mesh;

  explicit MeshAttrStorage(Mesh mesh) : mesh(std::move(mesh)) {}

  bool operator==(const KeyTy& key) const { return key == KeyTy(mesh); }

  static toolchain::hash_code hashKey(const KeyTy& key) {
    const Mesh& mesh = key;
    return toolchain::hash_value(mesh.ToString());
  }

  static MeshAttrStorage* construct(mlir::AttributeStorageAllocator& allocator,
                                    const KeyTy& key) {
    return new (allocator.allocate<MeshAttrStorage>()) MeshAttrStorage(key);
  }
  Mesh mesh;
};
}  // namespace detail

MeshAttr MeshAttr::get(MLIRContext* context, const Mesh& mesh) {
  return Base::get(context, mesh);
}

const MeshAttr::Mesh& MeshAttr::getValue() const { return getImpl()->mesh; }

// The storage class for LayoutAttr.
namespace detail {
struct LayoutAttrStorage : public AttributeStorage {
  using Layout = machina::dtensor::Layout;
  using KeyTy = Layout;

  explicit LayoutAttrStorage(Layout layout) : layout(std::move(layout)) {}

  bool operator==(const KeyTy& key) const { return key == KeyTy(layout); }

  static toolchain::hash_code hashKey(const KeyTy& key) {
    const Layout& layout = key;
    return toolchain::hash_value(layout.ToString());
  }

  static LayoutAttrStorage* construct(
      mlir::AttributeStorageAllocator& allocator, const KeyTy& key) {
    const Layout& layout = key;
    return new (allocator.allocate<LayoutAttrStorage>())
        LayoutAttrStorage(layout);
  }
  Layout layout;
};
}  // namespace detail

LayoutAttr LayoutAttr::get(mlir::MLIRContext* context,
                           machina::dtensor::Layout layout) {
  return Base::get(context, std::move(layout));
}

const LayoutAttr::Layout& LayoutAttr::getValue() const {
  return getImpl()->layout;
}

void DTensorDialect::registerAttributes() {
  addAttributes<dtensor::MeshAttr, dtensor::LayoutAttr>();
}

}  // namespace dtensor
}  // namespace mlir
