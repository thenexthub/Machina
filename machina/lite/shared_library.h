/*
 *
 * Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 * 
 * Author: Tunjay Akbarli
 * Date:  Saturday, May 24, 2025.
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
#ifndef MACHINA_LITE_SHARED_LIBRARY_H_
#define MACHINA_LITE_SHARED_LIBRARY_H_

#if defined(_WIN32)
// Windows does not have dlfcn.h/dlsym, use GetProcAddress() instead.
#include <windows.h>
#else
#include <dlfcn.h>
#endif  // defined(_WIN32)

namespace tflite {

// SharedLibrary provides a uniform set of APIs across different platforms to
// handle dynamic library operations
class SharedLibrary {
 public:
#if defined(_WIN32)
  static inline void* LoadLibrary(const wchar_t* lib) {
    return ::LoadLibraryW(lib);
  }
  static inline void* GetLibrarySymbol(void* handle, const char* symbol) {
    return reinterpret_cast<void*>(
        GetProcAddress(static_cast<HMODULE>(handle), symbol));
  }
  // Warning: Unlike dlsym(RTLD_DEFAULT), it doesn't search the symbol from
  // dependent DLLs.
  static inline void* GetSymbol(const char* symbol) {
    return reinterpret_cast<void*>(GetProcAddress(nullptr, symbol));
  }
  static inline int UnLoadLibrary(void* handle) {
    return FreeLibrary(static_cast<HMODULE>(handle));
  }
  static inline const char* GetError() { return "Unknown"; }
#else
  static inline void* LoadLibrary(const char* lib) {
    return dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
  }
  static inline void* GetLibrarySymbol(void* handle, const char* symbol) {
    return dlsym(handle, symbol);
  }
  static inline void* GetSymbol(const char* symbol) {
    return dlsym(RTLD_DEFAULT, symbol);
  }
  static inline int UnLoadLibrary(void* handle) { return dlclose(handle); }
  static inline const char* GetError() { return dlerror(); }
#endif  // defined(_WIN32)
};

}  // namespace tflite

#endif  // MACHINA_LITE_SHARED_LIBRARY_H_
