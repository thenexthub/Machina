/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
#ifndef MACHINA_PYTHON_FRAMEWORK_TEST_OPS_H_
#define MACHINA_PYTHON_FRAMEWORK_TEST_OPS_H_

#include "machina/core/framework/op_kernel.h"

namespace machina {

// Run a kernel on the GPU that sleeps for the given time
void GpuSleep(OpKernelContext* ctx, int seconds);

}  // namespace machina

#endif  // MACHINA_PYTHON_FRAMEWORK_TEST_OPS_H_
