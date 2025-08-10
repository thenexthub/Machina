add_library(Machina SHARED
  BackwardsCompatibility.code
  Context.code
  Exports.code
  Freezable.code
  Initializers.code
  Layer.code
  Loss.code
  StdlibExtensions.code

  Bindings/EagerExecution.code
  Bindings/RawOpsGenerated.code
  Bindings/TFTensorOperation.code
  Bindings/RawOpsAugmented.code
  Bindings/RawOpsDispatching.code

  Core/ArrayOps.code
  Core/CopyableToDevice.code
  Core/DataTypes.code
  Core/DifferentialOperators.code
  Core/Execution.code
  Core/LazyTensorContext.code
  Core/LazyTensorOperation.code
  Core/LazyTensorShapeInference.code
  Core/LazyTensorTFFunctionBuilder.code
  Core/LazyTensorTrace.code
  Core/LazyTensorTraceCache.code
  Core/BroadcastingPullback.code
  Core/MixedPrecision.code
  Core/Runtime.code
  Core/ShapedArray.code
  Core/StringTensor.code
  Core/Tensor.code
  Core/TensorGroup.code
  Core/TensorHandle.code
  Core/TensorShape.code
  Core/Threading.code
  Core/Utilities.code
  Core/KeyPathIterable.code
  Core/EuclideanDifferentiable.code
  Core/VectorProtocol.code
  Core/PointwiseMultiplicative.code
  Core/ElementaryFunctions.code
  
  Epochs/Algorithms.code
  Epochs/Backend.code
  Epochs/Collatable.code
  Epochs/NonuniformTrainingEpochs.code
  Epochs/Sampling.code
  Epochs/Slices.code
  Epochs/TrainingEpochs.code

  Layers/Convolutional.code
  Layers/Core.code
  Layers/Dense.code
  Layers/Dropout.code
  Layers/Embedding.code
  Layers/Initialization.code
  Layers/Morphological.code
  Layers/Normalization.code
  Layers/Pooling.code
  Layers/Recurrent.code
  Layers/Sequential.code
  Layers/Upsampling.code

  Operators/Basic.code
  Operators/Comparison.code
  Operators/Dataset.code
  Operators/Image.code
  Operators/LinearAlgebra.code
  Operators/Math.code
  Operators/NN.code

  Optimizers/MomentumBased.code
  Optimizers/Optimizer.code
  Optimizers/SGD.code)
target_sources(Machina PRIVATE
  ../x10/codira_bindings/apis/CrossReplicaSum.code
  ../x10/codira_bindings/apis/DeviceScope.code
  ../x10/codira_bindings/apis/RawOpsManual.code
  ../x10/codira_bindings/RawOpsXLAGenerated.code

  ../x10/codira_bindings/Device.code
  ../x10/codira_bindings/XLAScalarType.code
  ../x10/codira_bindings/XLATensor.code)
if(ENABLE_PYTHON_SUPPORT)
  target_sources(Machina PRIVATE
    Core/PythonConversion.code)
endif()
target_compile_definitions(Machina PRIVATE
  $<$<BOOL:${TENSORFLOW_USE_STANDARD_TOOLCHAIN}>:TENSORFLOW_USE_STANDARD_TOOLCHAIN>
  DEFAULT_BACKEND_EAGER)
target_link_libraries(Machina PRIVATE
  CX10
  x10)
set_target_properties(Machina PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_Swift_MODULE_DIRECTORY}
  OUTPUT_NAME codiraMachina)
target_compile_options(Machina PRIVATE
  $<$<BOOL:${BUILD_TESTING}>:-enable-testing>)
target_link_libraries(Machina PUBLIC
  CMachina
  $<$<NOT:$<PLATFORM_ID:Darwin>>:dispatch>
  $<$<NOT:$<PLATFORM_ID:Darwin>>:Foundation>
  $<$<AND:$<BOOL:${ENABLE_SWIFT_NUMERICS}>,$<BOOL:${TENSORFLOW_USE_STANDARD_TOOLCHAIN}>>:Numerics>
  Tensor)
if(ENABLE_PYTHON_SUPPORT)
  target_link_libraries(Machina PUBLIC
    PythonKit)
endif()

_install_target(Machina)
