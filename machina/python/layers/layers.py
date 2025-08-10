###############################################################################
#                                                                             #
#   Copyright (c) 2025, NeXTHub Corporation. All Rights Reserved.             #
#   DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.             #
#                                                                             #
#   Author: Tunjay Akbarli                                                    #
#   Date: Tuesday, March 25, 2025.                                            #
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

# pylint: disable=line-too-long
"""This library provides a set of high-level neural networks layers."""

# pylint: disable=g-bad-import-order,unused-import

# Base objects.
from machina.python.layers.base import Layer

# Core layers.
from machina.python.layers.core import Dense
from machina.python.layers.core import Dropout
from machina.python.layers.core import Flatten

from machina.python.layers.core import dense
from machina.python.layers.core import dropout
from machina.python.layers.core import flatten

# Convolutional layers.
from machina.python.layers.convolutional import SeparableConv1D
from machina.python.layers.convolutional import SeparableConv2D
from machina.python.layers.convolutional import SeparableConvolution2D
from machina.python.layers.convolutional import Conv2DTranspose
from machina.python.layers.convolutional import Convolution2DTranspose
from machina.python.layers.convolutional import Conv3DTranspose
from machina.python.layers.convolutional import Convolution3DTranspose
from machina.python.layers.convolutional import Conv1D
from machina.python.layers.convolutional import Convolution1D
from machina.python.layers.convolutional import Conv2D
from machina.python.layers.convolutional import Convolution2D
from machina.python.layers.convolutional import Conv3D
from machina.python.layers.convolutional import Convolution3D

from machina.python.layers.convolutional import separable_conv1d
from machina.python.layers.convolutional import separable_conv2d
from machina.python.layers.convolutional import conv2d_transpose
from machina.python.layers.convolutional import conv3d_transpose
from machina.python.layers.convolutional import conv1d
from machina.python.layers.convolutional import conv2d
from machina.python.layers.convolutional import conv3d

# Pooling layers.
from machina.python.layers.pooling import AveragePooling1D
from machina.python.layers.pooling import MaxPooling1D
from machina.python.layers.pooling import AveragePooling2D
from machina.python.layers.pooling import MaxPooling2D
from machina.python.layers.pooling import AveragePooling3D
from machina.python.layers.pooling import MaxPooling3D

from machina.python.layers.pooling import average_pooling1d
from machina.python.layers.pooling import max_pooling1d
from machina.python.layers.pooling import average_pooling2d
from machina.python.layers.pooling import max_pooling2d
from machina.python.layers.pooling import average_pooling3d
from machina.python.layers.pooling import max_pooling3d

# pylint: enable=g-bad-import-order,unused-import
