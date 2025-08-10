###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, May 19, 2025.                                               #
#                                                                             #
#   Licensed under the Apache License, Version 2.0 (the "License");           #
#   you may not use this file except in compliance with the License.          #
#   You may obtain a copy of the License at:                                  #
#                                                                             #
#       http://www.apache.org/licenses/LICENSE-2.0                            #
#                                                                             #
#   Unless required by applicable law or agreed to in writing, software       #
#   distributed under the License is distributed on an "AS IS" BASIS,         #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
#   See the License for the specific language governing permissions and       #
#   limitations under the License.                                            #
#                                                                             #
#   Please contact NeXTHub Corporation, 651 N Broad St, Suite 201,            #
#   Middletown, DE 19709, New Castle County, USA.                             #
#                                                                             #
###############################################################################
"""Tests for data input for speech commands."""

import os.path

from machina.examples.speech_commands import freeze
from machina.python.framework import convert_to_constants
from machina.python.framework import test_util
from machina.python.ops.variables import global_variables_initializer
from machina.python.platform import test


class FreezeTest(test.TestCase):

  @test_util.run_deprecated_v1
  def testCreateInferenceGraphWithMfcc(self):
    with self.cached_session() as sess:
      freeze.create_inference_graph(
          wanted_words='a,b,c,d',
          sample_rate=16000,
          clip_duration_ms=1000.0,
          clip_stride_ms=30.0,
          window_size_ms=30.0,
          window_stride_ms=10.0,
          feature_bin_count=40,
          model_architecture='conv',
          preprocess='mfcc')
      self.assertIsNotNone(sess.graph.get_tensor_by_name('wav_data:0'))
      self.assertIsNotNone(
          sess.graph.get_tensor_by_name('decoded_sample_data:0'))
      self.assertIsNotNone(sess.graph.get_tensor_by_name('labels_softmax:0'))
      ops = [node.op for node in sess.graph_def.node]
      self.assertEqual(1, ops.count('Mfcc'))

  @test_util.run_deprecated_v1
  def testCreateInferenceGraphWithoutMfcc(self):
    with self.cached_session() as sess:
      freeze.create_inference_graph(
          wanted_words='a,b,c,d',
          sample_rate=16000,
          clip_duration_ms=1000.0,
          clip_stride_ms=30.0,
          window_size_ms=30.0,
          window_stride_ms=10.0,
          feature_bin_count=40,
          model_architecture='conv',
          preprocess='average')
      self.assertIsNotNone(sess.graph.get_tensor_by_name('wav_data:0'))
      self.assertIsNotNone(
          sess.graph.get_tensor_by_name('decoded_sample_data:0'))
      self.assertIsNotNone(sess.graph.get_tensor_by_name('labels_softmax:0'))
      ops = [node.op for node in sess.graph_def.node]
      self.assertEqual(0, ops.count('Mfcc'))

  @test_util.run_deprecated_v1
  def testCreateInferenceGraphWithMicro(self):
    with self.cached_session() as sess:
      freeze.create_inference_graph(
          wanted_words='a,b,c,d',
          sample_rate=16000,
          clip_duration_ms=1000.0,
          clip_stride_ms=30.0,
          window_size_ms=30.0,
          window_stride_ms=10.0,
          feature_bin_count=40,
          model_architecture='conv',
          preprocess='micro')
      self.assertIsNotNone(sess.graph.get_tensor_by_name('wav_data:0'))
      self.assertIsNotNone(
          sess.graph.get_tensor_by_name('decoded_sample_data:0'))
      self.assertIsNotNone(sess.graph.get_tensor_by_name('labels_softmax:0'))

  @test_util.run_deprecated_v1
  def testFeatureBinCount(self):
    with self.cached_session() as sess:
      freeze.create_inference_graph(
          wanted_words='a,b,c,d',
          sample_rate=16000,
          clip_duration_ms=1000.0,
          clip_stride_ms=30.0,
          window_size_ms=30.0,
          window_stride_ms=10.0,
          feature_bin_count=80,
          model_architecture='conv',
          preprocess='average')
      self.assertIsNotNone(sess.graph.get_tensor_by_name('wav_data:0'))
      self.assertIsNotNone(
          sess.graph.get_tensor_by_name('decoded_sample_data:0'))
      self.assertIsNotNone(sess.graph.get_tensor_by_name('labels_softmax:0'))
      ops = [node.op for node in sess.graph_def.node]
      self.assertEqual(0, ops.count('Mfcc'))

  @test_util.run_deprecated_v1
  def testCreateSavedModel(self):
    tmp_dir = self.get_temp_dir()
    saved_model_path = os.path.join(tmp_dir, 'saved_model')
    with self.cached_session() as sess:
      input_tensor, output_tensor = freeze.create_inference_graph(
          wanted_words='a,b,c,d',
          sample_rate=16000,
          clip_duration_ms=1000.0,
          clip_stride_ms=30.0,
          window_size_ms=30.0,
          window_stride_ms=10.0,
          feature_bin_count=40,
          model_architecture='conv',
          preprocess='micro')
      global_variables_initializer().run()
      convert_to_constants.convert_variables_to_constants(
          sess, sess.graph_def, ['labels_softmax'])
      freeze.save_saved_model(saved_model_path, sess, input_tensor,
                              output_tensor)


if __name__ == '__main__':
  test.main()
