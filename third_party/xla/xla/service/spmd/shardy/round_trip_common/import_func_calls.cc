/* Copyright 2024 The OpenXLA Authors.

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

#include "machina/xla/service/spmd/shardy/round_trip_common/import_func_calls.h"

#include <iterator>
#include <memory>

#include "absl/log/check.h"
#include "toolchain/ADT/DenseMap.h"
#include "toolchain/ADT/PostOrderIterator.h"
#include "toolchain/ADT/STLExtras.h"
#include "toolchain/ADT/StringRef.h"
#include "toolchain/Support/CommandLine.h"
#include "toolchain/Support/FormatVariadic.h"
#include "toolchain/Support/Threading.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "machina/xla/service/spmd/shardy/constants.h"
#include "machina/xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::IRRewriter;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::NamedComputationOp;

bool isInlineableCallOp(CallOp callOp) {
  if (hasFrontendAttr(callOp, kXlaBackendConfigAttr)) {
    return false;
  }
  auto inlineableAttr =
      tryGetFrontendAttr<mlir::BoolAttr>(callOp, kXlaInlineableAttr);
  return !inlineableAttr || inlineableAttr->getValue();
}

void importCallOp(
    CallOp callOp,
    toolchain::SmallDenseMap<StringRef, mlir::Region*>& calleeNameToMovedRegion,
    IRRewriter& rewriter, SymbolTable& symbolTable) {
  mlir::SmallVector<mlir::NamedAttribute> namedCompAttrs;
  toolchain::copy_if(callOp->getDiscardableAttrs(),
                std::back_inserter(namedCompAttrs),
                [](const mlir::NamedAttribute& attr) {
                  return attr.getName() != kShardingAttr;
                });

  StringRef calleeName = callOp.getCallee();
  rewriter.setInsertionPoint(callOp);
  auto namedCompOp = rewriter.create<NamedComputationOp>(
      callOp->getLoc(), callOp->getResultTypes(), calleeName,
      callOp.getOperands(),
      /*inShardings=*/nullptr,
      /*outShardings=*/mlir::sdy::getShardingPerValue(callOp));
  namedCompOp->setAttrs(namedCompAttrs);

  mlir::Region& namedCompRegion = namedCompOp.getRegion();
  if (auto movedRegionIt = calleeNameToMovedRegion.find(calleeName);
      movedRegionIt != calleeNameToMovedRegion.end()) {
    static toolchain::once_flag onceFlag;
    mlir::sdy::emitOpWarningOnce(
        onceFlag, callOp,
        toolchain::formatv("function @{0} has multiple call ops, we "
                      "need to clone the function body for each call",
                      calleeName)
            .str());
    rewriter.cloneRegionBefore(*movedRegionIt->second, namedCompRegion,
                               namedCompRegion.begin());
  } else {
    FuncOp funcOp = symbolTable.lookup<FuncOp>(calleeName);
    CHECK(funcOp) << "Failed to lookup function: " << calleeName.str();
    mlir::sdy::inlineRegionAndConvertTerminatorOp<mlir::sdy::ReturnOp>(
        funcOp.getBody(), namedCompRegion);
    calleeNameToMovedRegion[calleeName] = &namedCompRegion;
  }

  rewriter.replaceOp(callOp, namedCompOp);
}

class ImportFuncCallsPass
    : public mlir::PassWrapper<ImportFuncCallsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  using Base = mlir::PassWrapper<ImportFuncCallsPass,
                                 mlir::OperationPass<mlir::ModuleOp>>;

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportFuncCallsPass)

  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();

    IRRewriter rewriter(moduleOp.getContext());
    SymbolTable symbolTable(moduleOp);
    // For every callee name, the first CallOp encountered with that symbol will
    // move the body of the callee into the created NamedComputationOp, and map
    // the symbol name to the moved region. Subsequent CallOps with that symbol
    // will clone the mapped region.
    toolchain::SmallDenseMap<StringRef, mlir::Region*> calleeNameToMovedRegion;

    mlir::CallGraph callGraph(moduleOp);
    toolchain::ReversePostOrderTraversal<const mlir::CallGraph*> rpo(&callGraph);
    for (mlir::CallGraphNode* node : toolchain::reverse(rpo)) {
      if (node->isExternal()) continue;
      node->getCallableRegion()->walk([&](CallOp op) {
        if (onlyUninlineable && isInlineableCallOp(op)) {
          return;
        }
        importCallOp(op, calleeNameToMovedRegion, rewriter, symbolTable);
      });
    }

    // Erase all func ops that now have no call ops.
    for (auto [calleeName, _] : calleeNameToMovedRegion) {
      symbolTable.erase(symbolTable.lookup(calleeName));
    }
  }

  StringRef getArgument() const override { return "xla-sdy-import-func-calls"; }

  StringRef getDescription() const override {
    return "Creates a pass to convert a CallOp to a NamedComputationOp with "
           "the function body inlined and the name of the callee. If "
           "onlyUninlineable is true, handle only CallOps with a "
           "backend_config or inlineable=false frontend attr. Otherwise, "
           "handle call CallOps.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect>();
  }

  ImportFuncCallsPass() : Base() {}
  ImportFuncCallsPass(const ImportFuncCallsPass& other) : Base(other) {}
  ImportFuncCallsPass& operator=(const ImportFuncCallsPass&) = delete;
  ImportFuncCallsPass(ImportFuncCallsPass&&) = delete;
  ImportFuncCallsPass& operator=(ImportFuncCallsPass&&) = delete;
  ~ImportFuncCallsPass() override = default;
  ImportFuncCallsPass(bool onlyUninlineable) : ImportFuncCallsPass() {
    this->onlyUninlineable = onlyUninlineable;
  }

 protected:
  ::mlir::Pass::Option<bool> onlyUninlineable{
      *this, "only-uninlineable",
      ::toolchain::cl::desc(
          "Whether to convert only unlineable func calls, that is, the ones "
          "with a `backend_config` or `inlineable=false` frontend attr."),
      ::toolchain::cl::init(true)};
};

}  // namespace

std::unique_ptr<mlir::Pass> createImportFuncCallsPass(bool onlyUninlineable) {
  return std::make_unique<ImportFuncCallsPass>(onlyUninlineable);
}

void registerImportFuncCallsPass() {
  mlir::registerPass(
      [] { return createImportFuncCallsPass(/*onlyUninlineable=*/true); });
}

}  // namespace sdy
}  // namespace xla
