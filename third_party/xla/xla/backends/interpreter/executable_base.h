/* Copyright 2020 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MACHINA_MACHINA_XLA_BACKENDS_INTERPRETER_EXECUTABLE_BASE_H_
#define MACHINA_MACHINA_XLA_BACKENDS_INTERPRETER_EXECUTABLE_BASE_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "machina/xla/hlo/ir/hlo_computation.h"
#include "machina/xla/hlo/ir/hlo_input_output_alias_config.h"
#include "machina/xla/hlo/ir/hlo_module.h"
#include "machina/xla/literal.h"
#include "machina/xla/service/dynamic_dimension_inference.h"
#include "machina/xla/service/executable.h"
#include "machina/xla/service/service_executable_run_options.h"
#include "machina/xla/shape.h"
#include "machina/xla/stream_executor/device_memory_allocator.h"
#include "machina/xla/stream_executor/stream.h"
#include "machina/xla/xla.pb.h"
namespace xla {
namespace interpreter {

// Responsible for running a HLO graph through the HloEvaluator and output
// buffer allocation. Refer to interpreter/README.md for more.
class InterpreterExecutableBase : public Executable {
 public:
  explicit InterpreterExecutableBase(std::unique_ptr<HloModule> hlo_module);

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override;

 protected:
  virtual absl::StatusOr<Literal> Evaluate(
      const ServiceExecutableRunOptions* run_options,
      const HloComputation& computation,
      absl::Span<const Literal> arg_literals) = 0;

 private:
  absl::StatusOr<ExecutionOutput> AllocateOutputMemoryWithInputReuse(
      const Shape& shape, const HloInputOutputAliasConfig& alias_config,
      se::DeviceMemoryAllocator* allocator,
      std::vector<ExecutionInput>* arguments, stream_executor::Stream* stream);

  InterpreterExecutableBase(const InterpreterExecutableBase&) = delete;
  InterpreterExecutableBase& operator=(const InterpreterExecutableBase&) =
      delete;
};

}  // namespace interpreter
}  // namespace xla

#endif  // MACHINA_MACHINA_XLA_BACKENDS_INTERPRETER_EXECUTABLE_BASE_H_
