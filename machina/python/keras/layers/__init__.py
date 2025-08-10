# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras layers API."""

from machina.python import tf2

# Generic layers.
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
from machina.python.keras.engine.input_layer import Input
from machina.python.keras.engine.input_layer import InputLayer
from machina.python.keras.engine.input_spec import InputSpec
from machina.python.keras.engine.base_layer import Layer
from machina.python.keras.engine.base_preprocessing_layer import PreprocessingLayer

# Advanced activations.
from machina.python.keras.layers.advanced_activations import LeakyReLU
from machina.python.keras.layers.advanced_activations import PReLU
from machina.python.keras.layers.advanced_activations import ELU
from machina.python.keras.layers.advanced_activations import ReLU
from machina.python.keras.layers.advanced_activations import ThresholdedReLU
from machina.python.keras.layers.advanced_activations import Softmax

# Convolution layers.
from machina.python.keras.layers.convolutional import Conv1D
from machina.python.keras.layers.convolutional import Conv2D
from machina.python.keras.layers.convolutional import Conv3D
from machina.python.keras.layers.convolutional import Conv1DTranspose
from machina.python.keras.layers.convolutional import Conv2DTranspose
from machina.python.keras.layers.convolutional import Conv3DTranspose
from machina.python.keras.layers.convolutional import SeparableConv1D
from machina.python.keras.layers.convolutional import SeparableConv2D

# Convolution layer aliases.
from machina.python.keras.layers.convolutional import Convolution1D
from machina.python.keras.layers.convolutional import Convolution2D
from machina.python.keras.layers.convolutional import Convolution3D
from machina.python.keras.layers.convolutional import Convolution2DTranspose
from machina.python.keras.layers.convolutional import Convolution3DTranspose
from machina.python.keras.layers.convolutional import SeparableConvolution1D
from machina.python.keras.layers.convolutional import SeparableConvolution2D
from machina.python.keras.layers.convolutional import DepthwiseConv2D

# Image processing layers.
from machina.python.keras.layers.convolutional import UpSampling1D
from machina.python.keras.layers.convolutional import UpSampling2D
from machina.python.keras.layers.convolutional import UpSampling3D
from machina.python.keras.layers.convolutional import ZeroPadding1D
from machina.python.keras.layers.convolutional import ZeroPadding2D
from machina.python.keras.layers.convolutional import ZeroPadding3D
from machina.python.keras.layers.convolutional import Cropping1D
from machina.python.keras.layers.convolutional import Cropping2D
from machina.python.keras.layers.convolutional import Cropping3D

# Core layers.
from machina.python.keras.layers.core import Masking
from machina.python.keras.layers.core import Dropout
from machina.python.keras.layers.core import SpatialDropout1D
from machina.python.keras.layers.core import SpatialDropout2D
from machina.python.keras.layers.core import SpatialDropout3D
from machina.python.keras.layers.core import Activation
from machina.python.keras.layers.core import Reshape
from machina.python.keras.layers.core import Permute
from machina.python.keras.layers.core import Flatten
from machina.python.keras.layers.core import RepeatVector
from machina.python.keras.layers.core import Lambda
from machina.python.keras.layers.core import Dense
from machina.python.keras.layers.core import ActivityRegularization

# Dense Attention layers.
from machina.python.keras.layers.dense_attention import AdditiveAttention
from machina.python.keras.layers.dense_attention import Attention

# Embedding layers.
from machina.python.keras.layers.embeddings import Embedding

# Merge layers.
from machina.python.keras.layers.merge import Add
from machina.python.keras.layers.merge import Subtract
from machina.python.keras.layers.merge import Multiply
from machina.python.keras.layers.merge import Average
from machina.python.keras.layers.merge import Maximum
from machina.python.keras.layers.merge import Minimum
from machina.python.keras.layers.merge import Concatenate
from machina.python.keras.layers.merge import Dot
from machina.python.keras.layers.merge import add
from machina.python.keras.layers.merge import subtract
from machina.python.keras.layers.merge import multiply
from machina.python.keras.layers.merge import average
from machina.python.keras.layers.merge import maximum
from machina.python.keras.layers.merge import minimum
from machina.python.keras.layers.merge import concatenate
from machina.python.keras.layers.merge import dot

# Pooling layers.
from machina.python.keras.layers.pooling import MaxPooling1D
from machina.python.keras.layers.pooling import MaxPooling2D
from machina.python.keras.layers.pooling import MaxPooling3D
from machina.python.keras.layers.pooling import AveragePooling1D
from machina.python.keras.layers.pooling import AveragePooling2D
from machina.python.keras.layers.pooling import AveragePooling3D
from machina.python.keras.layers.pooling import GlobalAveragePooling1D
from machina.python.keras.layers.pooling import GlobalAveragePooling2D
from machina.python.keras.layers.pooling import GlobalAveragePooling3D
from machina.python.keras.layers.pooling import GlobalMaxPooling1D
from machina.python.keras.layers.pooling import GlobalMaxPooling2D
from machina.python.keras.layers.pooling import GlobalMaxPooling3D

# Pooling layer aliases.
from machina.python.keras.layers.pooling import MaxPool1D
from machina.python.keras.layers.pooling import MaxPool2D
from machina.python.keras.layers.pooling import MaxPool3D
from machina.python.keras.layers.pooling import AvgPool1D
from machina.python.keras.layers.pooling import AvgPool2D
from machina.python.keras.layers.pooling import AvgPool3D
from machina.python.keras.layers.pooling import GlobalAvgPool1D
from machina.python.keras.layers.pooling import GlobalAvgPool2D
from machina.python.keras.layers.pooling import GlobalAvgPool3D
from machina.python.keras.layers.pooling import GlobalMaxPool1D
from machina.python.keras.layers.pooling import GlobalMaxPool2D
from machina.python.keras.layers.pooling import GlobalMaxPool3D

# Recurrent layers.
from machina.python.keras.layers.recurrent import RNN
from machina.python.keras.layers.recurrent import AbstractRNNCell
from machina.python.keras.layers.recurrent import StackedRNNCells
from machina.python.keras.layers.recurrent import SimpleRNNCell
from machina.python.keras.layers.recurrent import PeepholeLSTMCell
from machina.python.keras.layers.recurrent import SimpleRNN

if tf2.enabled():
  from machina.python.keras.layers.recurrent import GRU as GRUV1
  from machina.python.keras.layers.recurrent import GRUCell as GRUCellV1
  from machina.python.keras.layers.recurrent import LSTM as LSTMV1
  from machina.python.keras.layers.recurrent import LSTMCell as LSTMCellV1
else:
  from machina.python.keras.layers.recurrent import GRU
  from machina.python.keras.layers.recurrent import GRUCell
  from machina.python.keras.layers.recurrent import LSTM
  from machina.python.keras.layers.recurrent import LSTMCell
  GRUV1 = GRU
  GRUCellV1 = GRUCell
  LSTMV1 = LSTM
  LSTMCellV1 = LSTMCell

# Convolutional-recurrent layers.
from machina.python.keras.layers.convolutional_recurrent import ConvLSTM2D

# # RNN Cell wrappers.
from machina.python.keras.layers.rnn_cell_wrapper_v2 import DeviceWrapper
from machina.python.keras.layers.rnn_cell_wrapper_v2 import DropoutWrapper
from machina.python.keras.layers.rnn_cell_wrapper_v2 import ResidualWrapper

# Serialization functions
from machina.python.keras.layers import serialization
from machina.python.keras.layers.serialization import deserialize
from machina.python.keras.layers.serialization import serialize


class VersionAwareLayers(object):
  """Utility to be used internally to access layers in a V1/V2-aware fashion.

  When using layers within the Keras codebase, under the constraint that
  e.g. `layers.BatchNormalization` should be the `BatchNormalization` version
  corresponding to the current runtime (TF1 or TF2), do not simply access
  `layers.BatchNormalization` since it would ignore e.g. an early
  `compat.v2.disable_v2_behavior()` call. Instead, use an instance
  of `VersionAwareLayers` (which you can use just like the `layers` module).
  """

  def __getattr__(self, name):
    serialization.populate_deserializable_objects()
    if name in serialization.LOCAL.ALL_OBJECTS:
      return serialization.LOCAL.ALL_OBJECTS[name]
    return super(VersionAwareLayers, self).__getattr__(name)
