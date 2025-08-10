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

#include "machina/examples/wav_to_spectrogram/wav_to_spectrogram.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "machina/cc/ops/audio_ops.h"
#include "machina/cc/ops/const_op.h"
#include "machina/cc/ops/image_ops.h"
#include "machina/cc/ops/standard_ops.h"
#include "machina/core/framework/graph.pb.h"
#include "machina/core/framework/tensor.h"
#include "machina/core/framework/types.pb.h"
#include "machina/core/graph/default_device.h"
#include "machina/core/graph/graph_def_builder.h"
#include "machina/core/lib/core/errors.h"
#include "machina/core/lib/core/stringpiece.h"
#include "machina/core/lib/core/threadpool.h"
#include "machina/core/lib/io/path.h"
#include "machina/core/lib/strings/stringprintf.h"
#include "machina/core/platform/logging.h"
#include "machina/core/platform/types.h"
#include "machina/core/public/session.h"

using machina::DT_FLOAT;
using machina::DT_UINT8;
using machina::Output;
using machina::TensorShape;

// Runs a TensorFlow graph to convert an audio file into a visualization.
absl::Status WavToSpectrogram(const machina::string& input_wav,
                              int32_t window_size, int32_t stride,
                              float brightness,
                              const machina::string& output_image) {
  auto root = machina::Scope::NewRootScope();
  using namespace machina::ops;  // NOLINT(build/namespaces)
  // The following block creates a TensorFlow graph that:
  //  - Reads and decodes the audio file into a tensor of float samples.
  //  - Creates a float spectrogram from those samples.
  //  - Scales, clamps, and converts that spectrogram to 0 to 255 uint8's.
  //  - Reshapes the tensor so that it's [height, width, 1] for imaging.
  //  - Encodes it as a PNG stream and saves it out to a file.
  Output file_reader =
      machina::ops::ReadFile(root.WithOpName("input_wav"), input_wav);
  DecodeWav wav_decoder =
      DecodeWav(root.WithOpName("wav_decoder"), file_reader);
  Output spectrogram = AudioSpectrogram(root.WithOpName("spectrogram"),
                                        wav_decoder.audio, window_size, stride);
  Output brightness_placeholder =
      Placeholder(root.WithOpName("brightness_placeholder"), DT_FLOAT,
                  Placeholder::Attrs().Shape(TensorShape({})));
  Output mul = Mul(root.WithOpName("mul"), spectrogram, brightness_placeholder);
  Output min_const = Const(root.WithOpName("min_const"), 255.0f);
  Output min = Minimum(root.WithOpName("min"), mul, min_const);
  Output cast = Cast(root.WithOpName("cast"), min, DT_UINT8);
  Output expand_dims_const = Const(root.WithOpName("expand_dims_const"), -1);
  Output expand_dims =
      ExpandDims(root.WithOpName("expand_dims"), cast, expand_dims_const);
  Output squeeze = Squeeze(root.WithOpName("squeeze"), expand_dims,
                           Squeeze::Attrs().Axis({0}));
  Output png_encoder = EncodePng(root.WithOpName("png_encoder"), squeeze);
  machina::ops::WriteFile file_writer = machina::ops::WriteFile(
      root.WithOpName("output_image"), output_image, png_encoder);
  machina::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  // Build a session object from this graph definition. The power of TensorFlow
  // is that you can reuse complex computations like this, so usually we'd run a
  // lot of different inputs through it. In this example, we're just doing a
  // one-off run, so we'll create it and then use it immediately.
  std::unique_ptr<machina::Session> session(
      machina::NewSession(machina::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));

  // We're passing in the brightness as an input, so create a tensor to hold the
  // value.
  machina::Tensor brightness_tensor(DT_FLOAT, TensorShape({}));
  brightness_tensor.scalar<float>()() = brightness;

  // Run the session to analyze the audio and write out the file.
  TF_RETURN_IF_ERROR(
      session->Run({{"brightness_placeholder", brightness_tensor}}, {},
                   {"output_image"}, nullptr));
  return absl::OkStatus();
}
