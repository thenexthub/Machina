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
from machina.python.platform import googletest  # pylint: disable=g-direct-machina-import
from machina.python.util import pywrap_xla_ops


class XlaOpsetUtilsTest(googletest.TestCase):

  def testGetGpuCompilableKernelNames(self):
    """Tests retrieving compilable op names for GPU."""
    op_names = pywrap_xla_ops.get_gpu_kernel_names()
    self.assertGreater(op_names.__len__(), 0)
    self.assertEqual(op_names.count('Max'), 1)
    self.assertEqual(op_names.count('Min'), 1)
    self.assertEqual(op_names.count('MatMul'), 1)

  def testGetCpuCompilableKernelNames(self):
    """Tests retrieving compilable op names for CPU."""
    op_names = pywrap_xla_ops.get_cpu_kernel_names()
    self.assertGreater(op_names.__len__(), 0)
    self.assertEqual(op_names.count('Max'), 1)
    self.assertEqual(op_names.count('Min'), 1)
    self.assertEqual(op_names.count('MatMul'), 1)


if __name__ == '__main__':
  googletest.main()
