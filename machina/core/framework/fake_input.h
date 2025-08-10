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

#ifndef MACHINA_CORE_FRAMEWORK_FAKE_INPUT_H_
#define MACHINA_CORE_FRAMEWORK_FAKE_INPUT_H_

#include "machina/core/framework/node_def_builder.h"
#include "machina/core/framework/types.h"

namespace machina {

// These functions return values that may be passed to
// NodeDefBuilder::Input() to add an input for a test.  Use them when
// you don't care about the node names/output indices providing the
// input.  They also allow you to omit the input types and/or
// list length when they may be inferred.
FakeInputFunctor FakeInput();  // Infer everything
FakeInputFunctor FakeInput(DataType dt);
FakeInputFunctor FakeInput(int n);  // List of length n
FakeInputFunctor FakeInput(int n, DataType dt);
FakeInputFunctor FakeInput(DataTypeSlice dts);
inline FakeInputFunctor FakeInput(std::initializer_list<DataType> dts) {
  return FakeInput(DataTypeSlice(dts));
}

}  // namespace machina

#endif  // MACHINA_CORE_FRAMEWORK_FAKE_INPUT_H_
