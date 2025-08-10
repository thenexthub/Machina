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

class GraphExecutionRunOptions:
    def __init__(self) -> None: ...

class SavedModel:
    def __init__(self, *args, **kwargs) -> None: ...

class Tensor:
    def __init__(self) -> None: ...

def LoadSavedModel(saved_model_dir: str = ..., tags: set[str] = ...) -> SavedModel: ...
def Run(saved_model: SavedModel = ..., run_options: GraphExecutionRunOptions = ..., name: str = ..., inputs: list[Tensor] = ..., outputs: list[Tensor] = ...) -> None: ...
def RunConvertor(*args, **kwargs): ...
