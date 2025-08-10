/* Copyright 2017 The OpenXLA Authors.

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

#ifndef MACHINA_MACHINA_XLA_SERVICE_CPU_IR_FUNCTION_H_
#define MACHINA_MACHINA_XLA_SERVICE_CPU_IR_FUNCTION_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "toolchain/IR/Function.h"
#include "toolchain/IR/IRBuilder.h"
#include "toolchain/IR/Module.h"
#include "toolchain/IR/Value.h"
#include "machina/xla/service/cpu/ir_emission_utils.h"
#include "machina/xla/service/hlo_module_config.h"
#include "machina/xla/shape_util.h"
#include "machina/xla/types.h"

namespace xla {
namespace cpu {

// IrFunction creates and encapsulates an toolchain::Function, exposing methods to
// emitters for function and function argument access.
// The toolchain::Function is created with the standard function signature
// used in the XLA CPU backend (see ir_function.cc for argument details).
// In addition IrFunction saves the callers IR insert point during construction,
// and restores it after destruction.
//
// Example usage:
//
//    // Create and initialize new IrFunction.
//    std::unique_ptr<IrFunction> compute_function(new IrFunction(...));
//    // Emit IR for function body using IrFunction helper methods.
//    ...
//    // Store reference to toolchain::Function for future invocation.
//    ir_functions.push_back(compute_function.function());
//    // Delete IrFunction (finalizes IR function and restores caller insertion
//    // point).
//    compute_function.reset();
//

class IrFunction {
 public:
  IrFunction(const std::string& function_name,
             toolchain::Function::LinkageTypes linkage,
             const HloModuleConfig& module_config, toolchain::Module* llvm_module,
             toolchain::IRBuilderBase* b, int64_t num_dynamic_loop_bounds);

  // Initialize an toolchain::Function with existing function, created somewhere
  // else, omit any extra work.
  IrFunction(toolchain::IRBuilderBase* b, toolchain::Module* llvm_module,
             int64_t num_dynamic_loop_bounds, toolchain::Function* function,
             // Function argument IR values.
             // toolchain::Argument* result_arg, toolchain::Value* exec_run_options_arg,
             // toolchain::Value* parameters_arg, toolchain::Value* buffer_table_arg,
             toolchain::Value* dynamic_loop_bounds_arg,
             // toolchain::Value* profile_counters_arg, toolchain::Value* status_arg,
             //  Basic block containing return.
             toolchain::BasicBlock* return_block);

  ~IrFunction();

  // Emit IR to read and return the set of IR values representing the dynamic
  // loop bounds argument of this function. These bounds delimit the subset
  // of the output that will be written by the computation's root instruction at
  // runtime. This is used for parallel computations, where a single computation
  // is partitioned into N calls to a function with parallel loop bounds, and
  // then called N times in parallel with loop bounds limiting each call to
  // producing 1/N of the output.
  //
  // Each element in returned vector is a pair of ir values representing the
  // loop bounds for a specific dimension, where the first element of the pair
  // is the dimension start index, and the second element of the pair is the
  // dimension limit.
  //
  // EX: [dimension_i_index_start_ir_value, // dimension_i_index_limit_ir_value]
  DynamicLoopBounds GetDynamicLoopBounds();

  // Returns the encapculated toolchain::Function.
  toolchain::Function* function() { return function_; }

  // Get the toolchain::Value* that represents this functions "retval" argument.
  toolchain::Argument* result_arg() { return result_arg_; }

  // Get the xla::ExecutableRunOptions that represents this functions
  // "run_options" argument.
  toolchain::Value* exec_run_options_arg() { return exec_run_options_arg_; }

  // Get the toolchain::Value* that represents this functions parameters argument.
  toolchain::Value* parameters_arg() { return parameters_arg_; }

  // Get the toolchain::Value* that represents this functions "buffer_table"
  // argument.
  toolchain::Value* buffer_table_arg() { return buffer_table_arg_; }

  // Get the toolchain::Value* that represents this functions "prof_counters"
  // argument.
  toolchain::Value* profile_counters_arg() { return profile_counters_arg_; }

  // Get the toolchain::BasicBlock* that contains this function's "ret" instruction.
  toolchain::BasicBlock* return_block() { return return_block_; }

  // Get the toolchain::Value* that represents this function's "status" argument.
  toolchain::Value* status_arg() { return status_arg_; }

 private:
  // Initialize an toolchain::Function with standard signature based on arguments.
  void Initialize(const std::string& function_name,
                  toolchain::Function::LinkageTypes linkage,
                  const HloModuleConfig& module_config);

  // Emit ir to read and return the ir value for the dynamic loop bound at
  // 'offset' from the "dynamic_loop_bounds" argument of this function.
  toolchain::Value* GetDynamicLoopBound(int64_t offset);

  toolchain::IRBuilderBase* b_;
  toolchain::Module* llvm_module_;
  toolchain::IRBuilderBase::InsertPointGuard caller_insert_point_guard_;

  int64_t num_dynamic_loop_bounds_ = 0;
  // Encapsulated toolchain::Function.
  toolchain::Function* function_;
  // Function argument IR values.
  toolchain::Argument* result_arg_;
  toolchain::Value* exec_run_options_arg_;
  toolchain::Value* parameters_arg_;
  toolchain::Value* buffer_table_arg_;
  toolchain::Value* dynamic_loop_bounds_arg_ = nullptr;
  toolchain::Value* profile_counters_arg_;
  toolchain::Value* status_arg_;
  // Basic block containing return.
  toolchain::BasicBlock* return_block_;
};

// Returns arguments in `arguments` encoded as a single buffer, suitable for a
// function call.
toolchain::Value* EncodeArrayFunctionArguments(
    absl::Span<toolchain::Value* const> arguments, absl::string_view name,
    toolchain::IRBuilderBase* b);

// Returns an array of compute function call argument ir values.
std::vector<toolchain::Value*> GetArrayFunctionCallArguments(
    absl::Span<toolchain::Value* const> parameter_addresses, toolchain::IRBuilderBase* b,
    absl::string_view name, toolchain::Value* return_value_buffer,
    toolchain::Value* exec_run_options_arg, toolchain::Value* buffer_table_arg,
    toolchain::Value* status_arg, toolchain::Value* profile_counters_arg);

// Emits a call to a runtime fork/join function which dispatches parallel
// calls to 'parallel_function' (and joins threads before returning).
absl::Status EmitCallToParallelForkJoin(
    const std::vector<toolchain::Value*>& arguments, const Shape& shape,
    absl::Span<const int64_t> dimension_partition_counts,
    toolchain::IRBuilderBase* b, toolchain::Function* parallel_function,
    absl::string_view name);

}  // namespace cpu
}  // namespace xla

#endif  // MACHINA_MACHINA_XLA_SERVICE_CPU_IR_FUNCTION_H_
