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
#include "machina/core/data/service/snapshot/utils.h"

#include <memory>
#include <string>
#include <vector>

#include "machina/xla/tsl/platform/errors.h"
#include "machina/xla/tsl/platform/status.h"
#include "machina/xla/tsl/platform/test.h"
#include "machina/core/data/service/byte_size.h"
#include "machina/core/framework/dataset.pb.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor_shape.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/framework/variant.h"
#include "machina/core/framework/variant_encode_decode.h"
#include "tsl/platform/protobuf.h"

namespace machina {
namespace data {
namespace {

TEST(UtilsTest, EstimatedSizeBytes) {
  // int64 Tensor of size 1000.
  Tensor tensor(DT_INT64, TensorShape({10, 100}));
  std::vector<Tensor> Tensors{Tensor(DT_INT64, TensorShape({10, 100}))};
  EXPECT_GT(EstimatedSize(Tensors), ByteSize::Bytes(1000));
}

TEST(UtilsTest, EstimatedVariantSizeBytes) {
  // Variant Tensor of size 1000.
  std::unique_ptr<CompressedElement> compressed{
      protobuf::Arena::Create<CompressedElement>(nullptr)};
  compressed->set_data(std::string(1000, 'a'));
  Tensor tensor(DT_VARIANT, TensorShape({}));
  tensor.scalar<Variant>()() = *compressed;

  EXPECT_GT(EstimatedSize({tensor}), ByteSize::Bytes(1000));
}

TEST(UtilsTest, EstimatedMixedElementsSizeBytes) {
  // int64 Tensor of size 1000.
  Tensor int64_tensor(DT_INT64, TensorShape({10, 100}));

  // Variant Tensor of size 1000.
  std::unique_ptr<CompressedElement> compressed{
      protobuf::Arena::Create<CompressedElement>(nullptr)};
  compressed->set_data(std::string(1000, 'a'));
  Tensor variant_tensor(DT_VARIANT, TensorShape({}));
  variant_tensor.scalar<Variant>()() = *compressed;

  EXPECT_GT(EstimatedSize({int64_tensor, variant_tensor}),
            ByteSize::Bytes(2000));
}

TEST(UtilsTest, EmptyTensor) {
  EXPECT_GT(EstimatedSize({Tensor()}), ByteSize::Bytes(0));
}

}  // namespace
}  // namespace data
}  // namespace machina
