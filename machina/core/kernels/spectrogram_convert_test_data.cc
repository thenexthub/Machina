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

#include "machina/core/kernels/spectrogram_test_utils.h"

#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/status.h"
#include "machina/core/platform/init_main.h"
#include "machina/core/platform/logging.h"

namespace machina {
namespace wav {

// This takes a CSV file representing an array of complex numbers, and saves out
// a version using a binary format to save space in the repository.
absl::Status ConvertCsvToRaw(const string& input_filename) {
  std::vector<std::vector<std::complex<double>>> input_data;
  ReadCSVFileToComplexVectorOrDie(input_filename, &input_data);
  const string output_filename = input_filename + ".bin";
  if (!WriteComplexVectorToRawFloatFile(output_filename, input_data)) {
    return errors::InvalidArgument("Failed to write raw float file ",
                                   input_filename);
  }
  LOG(INFO) << "Wrote raw file to " << output_filename;
  return absl::OkStatus();
}

}  // namespace wav
}  // namespace machina

int main(int argc, char* argv[]) {
  machina::port::InitMain(argv[0], &argc, &argv);
  if (argc < 2) {
    LOG(ERROR) << "You must supply a CSV file as the first argument";
    return 1;
  }
  machina::string filename(argv[1]);
  absl::Status status = machina::wav::ConvertCsvToRaw(filename);
  if (!status.ok()) {
    LOG(ERROR) << "Error processing '" << filename << "':" << status;
    return 1;
  }
  return 0;
}
