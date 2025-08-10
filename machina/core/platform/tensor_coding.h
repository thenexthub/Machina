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

// Helper routines for encoding/decoding tensor contents.
#ifndef MACHINA_CORE_PLATFORM_TENSOR_CODING_H_
#define MACHINA_CORE_PLATFORM_TENSOR_CODING_H_

#include <string>

#include "machina/core/platform/platform.h"
#include "machina/core/platform/protobuf.h"
#include "machina/core/platform/refcount.h"
#include "machina/core/platform/stringpiece.h"
#include "machina/core/platform/types.h"

namespace machina {
namespace port {

// Store src contents in *out.  If backing memory for src is shared with *out,
// will ref obj during the call and will arrange to unref obj when no
// longer needed.
void AssignRefCounted(absl::string_view src, core::RefCounted* obj,
                      std::string* out);

// Copy contents of src to dst[0,src.size()-1].
inline void CopyToArray(const std::string& src, char* dst) {
  memcpy(dst, src.data(), src.size());
}

// Copy subrange [pos:(pos + n)) from src to dst. If pos >= src.size() the
// result is empty. If pos + n > src.size() the subrange [pos, size()) is
// copied.
inline void CopySubrangeToArray(const std::string& src, size_t pos, size_t n,
                                char* dst) {
  if (pos >= src.size()) return;
  memcpy(dst, src.data() + pos, std::min(n, src.size() - pos));
}

// Store encoding of strings[0..n-1] in *out.
void EncodeStringList(const tstring* strings, int64_t n, std::string* out);

// Decode n strings from src and store in strings[0..n-1].
// Returns true if successful, false on parse error.
bool DecodeStringList(const std::string& src, tstring* strings, int64_t n);

// Assigns base[0..bytes-1] to *s
void CopyFromArray(std::string* s, const char* base, size_t bytes);

// Encodes sequences of strings and serialized protocol buffers into a string.
// Normal usage consists of zero or more calls to Append() and a single call to
// Finalize().
class StringListEncoder {
 public:
  virtual ~StringListEncoder() = default;

  // Encodes the given protocol buffer. This may not be called after Finalize().
  virtual void Append(const protobuf::MessageLite& m) = 0;

  // Encodes the given string. This may not be called after Finalize().
  virtual void Append(const std::string& s) = 0;

  // Signals end of the encoding process. No other calls are allowed after this.
  virtual void Finalize() = 0;
};

// Decodes a string into sequences of strings (which may represent serialized
// protocol buffers). Normal usage involves a single call to ReadSizes() in
// order to retrieve the length of all the strings in the sequence. For each
// size returned a call to Data() is expected and will return the actual
// string.
class StringListDecoder {
 public:
  virtual ~StringListDecoder() = default;

  // Populates the given vector with the lengths of each string in the sequence
  // being decoded. Upon returning the vector is guaranteed to contain as many
  // elements as there are strings in the sequence.
  virtual bool ReadSizes(std::vector<uint32>* sizes) = 0;

  // Returns a pointer to the next string in the sequence, then prepares for the
  // next call by advancing 'size' characters in the sequence.
  virtual const char* Data(uint32 size) = 0;
};

std::unique_ptr<StringListEncoder> NewStringListEncoder(string* out);
std::unique_ptr<StringListDecoder> NewStringListDecoder(const string& in);

#if defined(MACHINA_PROTOBUF_USES_CORD)
// Store src contents in *out.  If backing memory for src is shared with *out,
// will ref obj during the call and will arrange to unref obj when no
// longer needed.
void AssignRefCounted(absl::string_view src, core::RefCounted* obj,
                      absl::Cord* out);

// TODO(kmensah): Macro guard this with a check for Cord support.
inline void CopyToArray(const absl::Cord& src, char* dst) {
  src.CopyToArray(dst);
}

// Copy n bytes of src to dst. If pos >= src.size() the result is empty.
// If pos + n > src.size() the subrange [pos, size()) is copied.
inline void CopySubrangeToArray(const absl::Cord& src, int64_t pos, int64_t n,
                                char* dst) {
  src.Subcord(pos, n).CopyToArray(dst);
}

// Store encoding of strings[0..n-1] in *out.
void EncodeStringList(const tstring* strings, int64_t n, absl::Cord* out);

// Decode n strings from src and store in strings[0..n-1].
// Returns true if successful, false on parse error.
bool DecodeStringList(const absl::Cord& src, std::string* strings, int64_t n);
bool DecodeStringList(const absl::Cord& src, tstring* strings, int64_t n);

// Assigns base[0..bytes-1] to *c
void CopyFromArray(absl::Cord* c, const char* base, size_t bytes);

std::unique_ptr<StringListEncoder> NewStringListEncoder(absl::Cord* out);
std::unique_ptr<StringListDecoder> NewStringListDecoder(const absl::Cord& in);
#endif  // defined(MACHINA_PROTOBUF_USES_CORD)

}  // namespace port
}  // namespace machina

#endif  // MACHINA_CORE_PLATFORM_TENSOR_CODING_H_
