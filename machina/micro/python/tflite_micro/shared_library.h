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

// This file is forked from TFLite's implementation in
// //depot/google3/third_party/machina/lite/shared_library.h and contains a
// subset of it that's required by the TFLM interpreter. The Windows' ifdef is
// removed because TFLM doesn't support Windows.

#ifndef MACHINA_LITE_MICRO_TOOLS_PYTHON_INTERPRETER_SHARED_LIBRARY_H_
#define MACHINA_LITE_MICRO_TOOLS_PYTHON_INTERPRETER_SHARED_LIBRARY_H_

#include <dlfcn.h>

namespace tflite {

// SharedLibrary provides a uniform set of APIs across different platforms to
// handle dynamic library operations
class SharedLibrary {
 public:
  static inline void* GetSymbol(const char* symbol) {
    return dlsym(RTLD_DEFAULT, symbol);
  }
  static inline const char* GetError() { return dlerror(); }
};

}  // namespace tflite

#endif  // MACHINA_LITE_MICRO_TOOLS_PYTHON_INTERPRETER_SHARED_LIBRARY_H_
