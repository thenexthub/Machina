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
#ifndef MACHINA_CORE_SUMMARY_SUMMARY_FILE_WRITER_H_
#define MACHINA_CORE_SUMMARY_SUMMARY_FILE_WRITER_H_

#include "absl/status/status.h"
#include "machina/core/kernels/summary_interface.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/env.h"
#include "machina/core/platform/types.h"

namespace machina {

/// \brief Creates SummaryWriterInterface which writes to a file.
///
/// The file is an append-only records file of tf.Event protos. That
/// makes this summary writer suitable for file systems like GCS.
///
/// It will enqueue up to max_queue summaries, and flush at least every
/// flush_millis milliseconds. The summaries will be written to the
/// directory specified by logdir and with the filename suffixed by
/// filename_suffix. The caller owns a reference to result if the
/// returned status is ok. The Env object must not be destroyed until
/// after the returned writer.
absl::Status CreateSummaryFileWriter(int max_queue, int flush_millis,
                                     const string& logdir,
                                     const string& filename_suffix, Env* env,
                                     SummaryWriterInterface** result);

}  // namespace machina

#endif  // MACHINA_CORE_SUMMARY_SUMMARY_FILE_WRITER_H_
