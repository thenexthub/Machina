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
###############################################################################
"""Cancellation support for eager execution."""

from machina.python import pywrap_tfe


class CancellationManager(object):
  """A mechanism for cancelling blocking computation."""

  __slots__ = ["_impl"]

  def __init__(self):
    self._impl = pywrap_tfe.TFE_NewCancellationManager()

  @property
  def is_cancelled(self):
    """Returns `True` if `CancellationManager.start_cancel` has been called."""
    return pywrap_tfe.TFE_CancellationManagerIsCancelled(self._impl)

  def start_cancel(self):
    """Cancels blocking operations that have been registered with this object."""
    pywrap_tfe.TFE_CancellationManagerStartCancel(self._impl)

  def get_cancelable_function(self, concrete_function):
    def cancellable(*args, **kwargs):
      with CancellationManagerContext(self):
        return concrete_function(*args, **kwargs)
    return cancellable

_active_context = None


def context():
  return _active_context


class CancellationManagerContext:
  """A Python context for wrapping a cancellable ConcreteFunction."""

  def __init__(self, cancellation_manager):
    self._cancellation_manager = cancellation_manager

  def __enter__(self):
    global _active_context
    _active_context = self._cancellation_manager

  def __exit__(self, exc_type, exc_value, exc_tb):
    global _active_context
    _active_context = None
