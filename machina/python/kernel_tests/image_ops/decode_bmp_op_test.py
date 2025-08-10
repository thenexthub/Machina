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
"""Tests for DecodeBmpOp."""

from machina.python.framework import constant_op
from machina.python.framework import dtypes
from machina.python.ops import array_ops
from machina.python.ops import image_ops
from machina.python.platform import test


class DecodeBmpOpTest(test.TestCase):

  def testex1(self):
    img_bytes = [[[0, 0, 255], [0, 255, 0]], [[255, 0, 0], [255, 255, 255]]]
    # Encoded BMP bytes from Wikipedia
    # BMP header bytes: https://en.wikipedia.org/wiki/List_of_file_signatures
    encoded_bytes = [
        0x42, 0x4d,
        0x46, 0, 0, 0,
        0, 0,
        0, 0,
        0x36, 0, 0, 0,
        0x28, 0, 0, 0,
        0x2, 0, 0, 0,
        0x2, 0, 0, 0,
        0x1, 0,
        0x18, 0,
        0, 0, 0, 0,
        0x10, 0, 0, 0,
        0x13, 0xb, 0, 0,
        0x13, 0xb, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0xff,
        0xff, 0xff, 0xff,
        0, 0,
        0xff, 0, 0,
        0, 0xff, 0,
        0, 0,
    ]

    byte_string = bytes(bytearray(encoded_bytes))
    img_in = constant_op.constant(byte_string, dtype=dtypes.string)
    decode = array_ops.squeeze(image_ops.decode_bmp(img_in))

    with self.cached_session():
      decoded = self.evaluate(decode)
      self.assertAllEqual(decoded, img_bytes)

  def testGrayscale(self):
    img_bytes = [[[255], [0]], [[255], [0]]]
    # BMP header bytes: https://en.wikipedia.org/wiki/List_of_file_signatures
    encoded_bytes = [
        0x42,
        0x4d,
        0x3d,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0x36,
        0,
        0,
        0,
        0x28,
        0,
        0,
        0,
        0x2,
        0,
        0,
        0,
        0x2,
        0,
        0,
        0,
        0x1,
        0,
        0x8,
        0,
        0,
        0,
        0,
        0,
        0x10,
        0,
        0,
        0,
        0x13,
        0xb,
        0,
        0,
        0x13,
        0xb,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0xff,
        0,
        0,
        0,
        0xff,
        0,
        0,
        0,
    ]

    byte_string = bytes(bytearray(encoded_bytes))
    img_in = constant_op.constant(byte_string, dtype=dtypes.string)
    # TODO(b/159600494): Currently, `decode_bmp` op does not validate input
    # magic bytes.
    decode = image_ops.decode_bmp(img_in)

    with self.cached_session():
      decoded = self.evaluate(decode)
      self.assertAllEqual(decoded, img_bytes)


if __name__ == "__main__":
  test.main()
