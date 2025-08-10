###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Saturday, May 31, 2025.                                             #
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
###############################################################################=
import os

from machina.python.framework import test_util
from machina.python.platform import test
from tflite_micro.machina.lite.micro.python.tflite_size.src import flatbuffer_size


class FlatbufferSizeTest(test_util.TensorFlowTestCase):

  def _compareFile(self, file1, data2):
    with open(file1, 'rb') as f1:
      data1 = f1.read()
    self.assertEqual(data1, data2.encode())

  def testCompareWithTFLite(self):
    root_dir = os.path.split(os.path.abspath(__file__))[0]

    in_filename = root_dir + '/simple_add_model.tflite'
    gold_json_file = root_dir + '/gold_simple_add_model_json.txt'
    gold_html_file = root_dir + '/gold_simple_add_model_html.txt'

    with open(in_filename, 'rb') as f:
      model = f.read()

    html_string, formatted_json = flatbuffer_size.convert_tflite_to_html(model)

    self._compareFile(gold_json_file, formatted_json)
    self._compareFile(gold_html_file, html_string)


if __name__ == '__main__':
  test.main()
