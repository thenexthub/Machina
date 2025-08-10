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
#include "machina/core/runtime_fallback/runtime/runtime_fallback_kernels.h"

#include <gtest/gtest.h>
#include "toolchain/ADT/SmallVector.h"
#include "machina/c/eager/abstract_tensor_handle.h"
#include "machina/core/framework/tensor_testutil.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/lib/core/status_test_util.h"
#include "machina/core/runtime_fallback/test/coreruntime_driver.h"
#include "tfrt/core_runtime/op_attrs.h"  // from @tf_runtime

namespace tfrt {
namespace {

TEST(RuntimeFallbackKernelsTest, CallEagerExecute) {
  auto driver = CoreRuntimeDriver();
  driver.InitializeCpuRuntimeFallbackOpHandler();

  auto exec_ctx = driver.CreateExecutionContext(__FILE__, __LINE__);

  machina::Tensor input(machina::DT_FLOAT, {2, 2});
  machina::test::FillValues<float>(&input, {1, 1, 1, 1});
  machina::TensorHandle* input_th =
      machina::TensorHandle::CreateLocalHandle(input);

  tfrt::OpAttrs matmul_attrs;
  matmul_attrs.Set<bool>("transpose_a", false);
  matmul_attrs.Set<bool>("transpose_b", false);
  tfrt::OpAttrsRef matmul_attrs_ref = matmul_attrs.freeze();

  toolchain::SmallVector<machina::AbstractTensorHandle*, 1> results;
  results.resize(1);

  auto eager_ctx_expected = machina::tfd::GetEagerContext(exec_ctx);
  // Assert there's no error obtaining EagerContext.
  ASSERT_FALSE(!eager_ctx_expected);
  machina::EagerContext* eager_ctx = eager_ctx_expected.get();

  TF_EXPECT_OK(machina::tfd::CallEagerExecute(
      exec_ctx, eager_ctx, "MatMul", /*device_name=*/"", {input_th, input_th},
      matmul_attrs_ref, results));

  ASSERT_EQ(results.size(), 1);

  machina::TensorHandle* res_th =
      machina::TensorHandleFromInterface(results[0]);
  const machina::Tensor* res_tensor;
  TF_EXPECT_OK(res_th->Tensor(&res_tensor));
  EXPECT_EQ(res_th->DataType(), machina::DT_FLOAT);
  int64_t num_elements;
  TF_EXPECT_OK(res_th->NumElements(&num_elements));
  EXPECT_EQ(num_elements, 4);

  machina::Tensor expected(machina::DT_FLOAT, {2, 2});
  machina::test::FillValues<float>(&expected, {2, 2, 2, 2});
  machina::test::ExpectTensorEqual<float>(*res_tensor, expected);

  // Deallocate TensorHandles.
  input_th->Unref();
  res_th->Unref();
}

}  // namespace
}  // namespace tfrt
