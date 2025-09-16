#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUOPTIMIZEWARPLAYOUT
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUOptimizeWarpLayoutPass
    : public impl::TritonGPUOptimizeWarpLayoutBase<
          TritonGPUOptimizeWarpLayoutPass> {
public:
  using impl::TritonGPUOptimizeWarpLayoutBase<
      TritonGPUOptimizeWarpLayoutPass>::TritonGPUOptimizeWarpLayoutBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    m.walk([&](triton::FuncOp funcOp) -> {
      LayoutPropagation layoutPropagation(funcOp);
      layoutPropagation.run();
    });

    findReductions(m);

  }
};

class LayoutPropagation {
  public:
  LayoutPropagation(FuncOp funcOp) : funcOp(funcOp) {}

  LogicalResult findReductions() {
    DenseSet<triton::ReduceOp> reduceOps;
    funcOp.walk([&](triton::ReduceOp reduceOp) -> void {
      reduceOps.insert(reduceOp);
    });

    // TODO: for each of the ReduceOps, insert an element into reduceAttributes in which the key is the input to the reduceOp and the value is the attribute on the reduceOp
    for (auto reduceOp : reduceOps) {
      if (reduceOp.getSrcs().size() != 1) {
        // Only consider reductions that take 1 input for simplicity
        return failure();
      }
      auto firstInput = reduceOp.getSrcs()[0];
      auto tensorType = firstInput.getType().dyn_cast<RankedTensorType>();
      if (!tensorType) {
        return failure();
      }
      if (auto encoding = tensorType.getEncoding()) {
        layouts[reduceOp] = encoding;
      } else {
        return failure();
      }
    }
      
    // TODO: iterate through reduceOps and assign the ideal layouts we want to assign

    return success();
  }

private:
  FuncOp funcOp;
  DenseMap<Value, Attribute> layouts;

  enum class AssignmentResult { Failure, NoOp, Success };

  /*
  LogicalResult propagateBlockedLayout(Operation *initOp,
                                       BlockedEncodingAttr oldLayout,
                                       BlockedEncodingAttr newLayout) {
    // Extract layouts from both operands and results
    SmallVector<Operation *> stack;
    stack.push_back(initOp);

    while (!stack.empty()) {
      Operation *op = stack.back();
      stack.pop_back();

      for (Value value : op->getOperands()) {
        auto state = assignNewLayout(value, oldLayout, newLayout);
        if (state == AssignmentResult::Failure) {
          return failure();
        } else if (state == AssignmentResult::Success) {
          // Only propagate if we successfully assigned a new layout
          for (Operation *user : value.getUsers()) {
            stack.push_back(user);
          }
        }
      }

      // Process results - propagate layout backwards to defining operations
      for (Value result : op->getResults()) {
        auto state = assignNewLayout(result, oldLayout, newLayout);
        if (state == AssignmentResult::Failure) {
          return failure();
        } else if (state == AssignmentResult::Success) {
          // Only propagate if we successfully assigned a new layout
          if (Operation *defOp = result.getDefiningOp()) {
            stack.push_back(defOp);
          }
        }
      }
    }

    return success();
  }
    */

  bool stopPropagation(Value value, Attribute attr) const {
    // TODO: prevent propagation of:
    //   * constants
    return true;
  }
  
  bool invalidPropagation(Value value, Attribute attr) const {
    // Non-blocked layouts, inputs, etc.
    return true;
  }

  AssignmentResult assignNewLayout(Value value, BlockedEncodingAttr oldLayout,
                                   BlockedEncodingAttr newLayout) {
    // Inline getNewLayout logic
    auto ty = value.getType();
    std::optional<Attribute> targetLayout;

    if (llvm::isa<RankedTensorType>(ty)) {
      auto tensorTy = llvm::cast<RankedTensorType>(ty);
      auto tensorEnc = tensorTy.getEncoding();
      if (auto blockedEnc = llvm::dyn_cast<BlockedEncodingAttr>(tensorEnc)) {
        if (blockedEnc.getSizePerThread() == oldLayout.getSizePerThread() &&
            blockedEnc.getWarpsPerCTA() == oldLayout.getWarpsPerCTA()) {
          targetLayout = newLayout;
        } else {
          return AssignmentResult::Failure;
        }
      } else {
        return AssignmentResult::Failure;
      }
    } else if (llvm::isa<IntegerType, PointerType>(ty)) {
      // TODO: Handle non-tensor types - for now return failure
      return AssignmentResult::NoOp;
    } else {
        return AssignmentResult::Failure;
    }


    if (!targetLayout.has_value()) {
      return AssignmentResult::Failure;
    }

    auto it = layouts.find(value);
    if (it != layouts.end()) {
      if (it->second != *targetLayout) {
        return AssignmentResult::Failure;
      }
      return AssignmentResult::NoOp;
    }

    layouts[value] = *targetLayout;
    return AssignmentResult::Success;
  }

  BlockedEncodingAttr
  getNewBlockedLayout(Value val, const BlockedEncodingAttr &oldLayout,
                      const BlockedEncodingAttr &newLayout) const {
    // TODO specifically implement the RankedTensorType / BlockedEncodingAttr
    // case so it can be reused for slicing
    return newLayout; // Placeholder implementation
  }

  RankedTensorType getNewRankedTensorType(Value val,
                                          const RankedTensorType &oldType) {
    auto oldEncoding = llvm::cast<BlockedEncodingAttr>(oldType.getEncoding());
    auto shape = oldType.getShape();
    // TODO: implement the rest of this method
    return oldType;
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
