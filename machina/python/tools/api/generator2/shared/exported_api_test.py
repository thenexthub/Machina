###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Monday, July 14, 2025.                                              #
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
from machina.python.platform import test
from machina.python.tools.api.generator2.shared import exported_api

_EXPORTS = exported_api.ExportedApi(
    docs=[
        exported_api.ExportedDoc(
            file_name="tf/python/framework/tensor.py",
            line_no=0,
            modules=("tf",),
            docstring="This is a docstring",
        ),
    ],
    symbols=[
        exported_api.ExportedSymbol(
            file_name="tf/python/framework/tensor.py",
            line_no=139,
            symbol_name="Tensor",
            v1_apis=("tf.Tensor",),
            v2_apis=(
                "tf.Tensor",
                "tf.experimental.numpy.ndarray",
            ),
        ),
        exported_api.ExportedSymbol(
            file_name="tf/python/framework/tensor.py",
            line_no=770,
            symbol_name="Tensor",
            v1_apis=("tf.enable_tensor_equality",),
            v2_apis=(),
        ),
    ],
)


class ExportedApiTest(test.TestCase):

  def test_read_write(self):
    filename = self.get_temp_dir() + "/test_write.json"
    _EXPORTS.write(filename)
    e = exported_api.ExportedApi()
    e.read(filename)

    self.assertEqual(e, _EXPORTS)


if __name__ == "__main__":
  test.main()
