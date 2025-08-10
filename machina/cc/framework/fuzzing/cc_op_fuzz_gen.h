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

#ifndef MACHINA_CC_FRAMEWORK_FUZZING_CC_OP_FUZZ_GEN_H_
#define MACHINA_CC_FRAMEWORK_FUZZING_CC_OP_FUZZ_GEN_H_

#include "machina/cc/framework/cc_op_gen_util.h"
#include "machina/core/framework/op_def.pb.h"
#include "machina/core/framework/op_gen_lib.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace cc_op {

// String with single fuzzer file content.
string WriteSingleFuzzer(const OpInfo& op_info, bool is_fuzzable);

// Do we have all we need to create a fuzzer
bool OpFuzzingIsOk(const OpInfo& op_info);

}  // namespace cc_op
}  // namespace machina

#endif  // MACHINA_CC_FRAMEWORK_FUZZING_CC_OP_FUZZ_GEN_H_
