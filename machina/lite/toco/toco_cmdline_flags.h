/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
#ifndef MACHINA_LITE_TOCO_TOCO_CMDLINE_FLAGS_H_
#define MACHINA_LITE_TOCO_TOCO_CMDLINE_FLAGS_H_

#include <string>
#include <vector>
#include "machina/lite/toco/args.h"
#include "machina/lite/toco/toco_flags.pb.h"
#include "machina/lite/toco/types.pb.h"

namespace toco {
// Parse and remove arguments handled from toco. Returns true if parsing
// is successful. msg has the usage string if there was an error or
// "--help" was specified
bool ParseTocoFlagsFromCommandLineFlags(int* argc, char* argv[],
                                        std::string* msg,
                                        ParsedTocoFlags* parsed_toco_flags_ptr);
// Populate the TocoFlags proto with parsed_toco_flags data.
void ReadTocoFlagsFromCommandLineFlags(const ParsedTocoFlags& parsed_toco_flags,
                                       TocoFlags* toco_flags);

}  // namespace toco

#endif  // MACHINA_LITE_TOCO_TOCO_CMDLINE_FLAGS_H_
