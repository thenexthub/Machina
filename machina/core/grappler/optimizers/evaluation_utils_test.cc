#include "machina/core/platform/cpu_info.h"
/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/ThreadPool"  // from @eigen_archive
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/tensor.pb.h"
#include "machina/core/grappler/optimizers/evaluation_utils.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/platform/test.h"

namespace machina {
namespace grappler {

TEST(EvaluationUtilsTest, DeviceSimple_BasicProperties) {
  DeviceSimple dsimple;
  ASSERT_TRUE(dsimple.has_eigen_cpu_device());
  const Eigen::ThreadPoolInterface* pool =
      dsimple.eigen_cpu_device()->getPool();
  ASSERT_NE(pool, nullptr);
}

TEST(EvaluationUtilsTest, DeviceSimple_MakeTensorFromProto) {
  DeviceSimple dsimple;

  TensorProto proto;
  Tensor tensor;
  EXPECT_FALSE(dsimple.MakeTensorFromProto(proto, {}, &tensor).ok());

  Tensor original(machina::DT_INT16, TensorShape{4, 2});
  original.flat<int16>().setRandom();

  original.AsProtoTensorContent(&proto);
  TF_ASSERT_OK(dsimple.MakeTensorFromProto(proto, {}, &tensor));

  ASSERT_EQ(tensor.dtype(), original.dtype());
  ASSERT_EQ(tensor.shape(), original.shape());

  auto buf0 = original.flat<int16>();
  auto buf1 = tensor.flat<int16>();
  ASSERT_EQ(buf0.size(), buf1.size());
  for (int i = 0; i < buf0.size(); ++i) {
    EXPECT_EQ(buf0(i), buf1(i));
  }
}
}  // namespace grappler
}  // namespace machina
