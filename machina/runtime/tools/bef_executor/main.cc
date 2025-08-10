// Copyright 2020 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- Host Executor Test Driver ------------------------------------------===//
//
// This file parses command-line options and runs a given mlir file using test
// driver library.

#include <optional>
#include <string>

#include "toolchain/Support/CommandLine.h"
#include "tfrt/bef_executor_driver/bef_executor_driver.h"
#include "tfrt/host_context/host_allocator.h"
#include "tfrt/tracing/tracing.h"

static toolchain::cl::opt<std::string> cl_input_filename(  // NOLINT
    toolchain::cl::Positional, toolchain::cl::desc("<input file>"), toolchain::cl::init("-"));

static toolchain::cl::list<std::string> cl_shared_libs(  // NOLINT
    "shared_libs", toolchain::cl::desc("Specify dynamic library with ops"),
    toolchain::cl::ZeroOrMore, toolchain::cl::MiscFlags::CommaSeparated);

static toolchain::cl::list<std::string> cl_functions(  // NOLINT
    "functions", toolchain::cl::desc("Specify MLIR functions to run"),
    toolchain::cl::ZeroOrMore, toolchain::cl::MiscFlags::CommaSeparated);

static toolchain::cl::opt<std::string> cl_test_init_function(  // NOLINT
    "test_init_function",
    toolchain::cl::desc(
        "Specify an init function that will be invoked as "
        "part of initialization, before invoking any other MLIR "
        "functions even if it is not specified in --functions flag."),
    toolchain::cl::init(""));

// Enable ConcurrentWorkQueue types to be specified on the command line.
static toolchain::cl::opt<std::string> cl_work_queue_type(  // NOLINT
    "work_queue_type",
    toolchain::cl::desc("Specify concurrent work queue type (s, mstd, ...):"),
    toolchain::cl::init("s"));

// Enable HostAllocator types to be specified on the command line.
static toolchain::cl::opt<tfrt::HostAllocatorType> cl_host_allocator_type(  // NOLINT
    "host_allocator_type", toolchain::cl::desc("Specify host allocator type:"),
    toolchain::cl::values(
        clEnumValN(tfrt::HostAllocatorType::kMalloc, "malloc", "Malloc."),
        clEnumValN(tfrt::HostAllocatorType::kTestFixedSizeMalloc,
                   "test_fixed_size_1k",
                   "Fixed size (1 kB) Malloc for testing."),
        clEnumValN(tfrt::HostAllocatorType::kProfiledMalloc,
                   "profiled_allocator", "Malloc with metric profiling."),
        clEnumValN(tfrt::HostAllocatorType::kLeakCheckMalloc,
                   "leak_check_allocator", "Malloc with memory leak check.")),
    toolchain::cl::init(tfrt::HostAllocatorType::kLeakCheckMalloc));

// Enable aggregate op handler types to be specified on the command line.
static toolchain::cl::opt<bool> cl_enable_tracing(  // NOLINT
    "enable_tracing", toolchain::cl::desc("Enable Performance Tracing"),
    toolchain::cl::Optional, toolchain::cl::ValueDisallowed);

static toolchain::cl::opt<tfrt::tracing::TracingLevel> cl_tracing_level(  // NOLINT
    "tracing_level", toolchain::cl::desc("Specify the Tracing Level:"),
    toolchain::cl::values(
        clEnumValN(tfrt::tracing::TracingLevel::Default, "default", "default"),
        clEnumValN(tfrt::tracing::TracingLevel::Verbose, "verbose", "verbose"),
        clEnumValN(tfrt::tracing::TracingLevel::Debug, "debug", "debug")),
    toolchain::cl::init(tfrt::tracing::TracingLevel::Default));

// Print error code if there's any error.
static toolchain::cl::opt<bool> cl_print_error_code(  // NOLINT
    "print_error_code",
    toolchain::cl::desc("Print error code if there's any error."),
    toolchain::cl::Optional, toolchain::cl::ValueDisallowed);

//===----------------------------------------------------------------------===//
// Driver main
//===----------------------------------------------------------------------===//

int main(int argc, char** argv) {
  toolchain::cl::ParseCommandLineOptions(argc, argv, "HostExecutor test driver\n");

  tfrt::RunBefConfig run_config;
  run_config.program_name = argv[0];
  run_config.input_filename = cl_input_filename;
  run_config.shared_libs = cl_shared_libs;
  run_config.functions = cl_functions;
  run_config.test_init_function = cl_test_init_function;
  run_config.work_queue_type = cl_work_queue_type;
  run_config.host_allocator_type = cl_host_allocator_type;
  run_config.print_error_code = cl_print_error_code;

  std::optional<tfrt::tracing::TracingRequester> tracing;
  if (cl_enable_tracing) tracing.emplace();
  tfrt::tracing::SetTracingLevel(cl_tracing_level);

  return RunBefExecutor(run_config);
}
