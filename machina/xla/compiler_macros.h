/* Copyright 2017 The OpenXLA Authors.

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

#ifndef MACHINA_XLACOMPILER_MACROS_H_
#define MACHINA_XLACOMPILER_MACROS_H_

#if (defined(__GNUC__) || defined(__clang__)) && defined(__SSE2__)
#define MACHINA_XLAHAS_SSE2
#elif defined(_MSC_VER) && !defined(_M_ARM64EC) && defined(_M_X64)
#define MACHINA_XLAHAS_SSE2
#elif defined(_MSC_VER) && !defined(_M_ARM64EC) && \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define MACHINA_XLAHAS_SSE2
#elif defined(__AVX__)
#define MACHINA_XLAHAS_SSE2
#endif

#if defined(_M_ARM64) || defined(_M_ARM64EC)
#define MACHINA_XLAHAS_ARM64
#define MACHINA_XLAHAS_ARM_NEON
#elif defined(__ARM_NEON) && !defined(__ARM_BIG_ENDIAN)
#define MACHINA_XLAHAS_ARM_NEON

#if defined(__aarch64__)
#define MACHINA_XLAHAS_ARM64
#endif  // defined(__aarch64__)

#endif  // defined(_M_ARM64) || defined(_M_ARM64EC)

#if defined(__clang__)
#define MACHINA_XLAUNROLL _Pragma("unroll")
#elif defined(__GNUC__)
#define MACHINA_XLAUNROLL _Pragma("GCC unroll 128")
#else
#define MACHINA_XLAUNROLL
#endif

#if defined(__GNUC__) || defined(__clang__)
#define MACHINA_XLAFLATTEN __attribute__((flatten))
#elif defined(_MSC_VER)
#define MACHINA_XLAFLATTEN [[msvc::flatten]]
#else
#define MACHINA_XLAFLATTEN
#endif

#endif  // MACHINA_XLACOMPILER_MACROS_H_
