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

    SmallVector<ReduceOp> reduceOps;
    m.walk([&](ReduceOp op) { reduceOps.push_back(op); });

    for (auto reduceOp : reduceOps) {
      auto rankedTensorType = reduceOp->getType().dyn_cast<RankedTensorType>();
      if (!rankedTensorType) {
        continue;
      }

      if (!rankedTensorType.getEncoding().isa<BlockedEncodingAttr>()) {
        continue;
      }

      BlockedLayoutAttribute newLayout = getNewBlockedLayout(reduceOp, rankedTensorType);
    }

    // TODO: Implement warp layout optimization
  }

private:
  DenseMap<Value, Type> layouts;

  enum class AssignmentResult {Failure, NoOp, Success};

  LogicalResult propagateBlockedLayout(Operation* initOp, BlockedLayoutAttribute oldLayout, BlockedLayoutAttribute newLayout) {
    // Extract layouts from both operands and results
    SmallVector<Operation*> stack;
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
  }

  AssignmentResult assignNewLayout(Value value, BlockedLayoutAttribute oldLayout, BlockedLayoutAttribute newLayout) {
    auto targetLayout = getNewLayout(value, oldLayout, newLayout);
    if (!targetLayout.has_value()) {
      return AssignmentResult::Failure;
    }

    if (auto alreadyAssigned = layouts.find(value)) {
      if (alreadyAssigned->second != targetLayout) {
        return AssignmentResult::Failure;
      }
      return AssignmentResult::NoOp;
    }

    layouts[value] = targetLayout;
    return AssignmentResult::Success;
  }

  std::optional<Attribute> getNewLayout(Value val, const BlockedLayoutAttribute& oldLayout, const BlockedLayoutAttribute& newLayout) const {
    auto ty = val.getType();

    if (ty.isa<RankedTensorType>()) {
      auto tensorTy = ty.cast<RankedTensorType>();
      auto tensorEnc = tensorTy.getEncoding();
      if (auto blockedEnc = tensorEnc.dyn_cast<BlockedEncodingAttr>()) {
        if (blockedEnc.getShape() == oldLayout.getShape() &&
            blockedEnc.getWarpsPerCTA() == oldLayout.getWarpsPerCTA()) {
          return newLayout;
        }
      }
    } else if (ty.isa<IntegerType, PointerType>()) {
      return ty;
    }
    return std::nullopt;
  }

  BlockedLayoutAttribute getNewBlockedLayout(Value val, const BlockedLayoutAttribute& oldLayout, const BlockedLayoutAttribute& newLayout) const {
    // TODO specifically implement the RankedTensorType / BlockedEncodingAttr case so it can be reused for slicing
  }

  RankedTensorType getNewRankedTensorType(Value val, const RankedTensorType& oldType) {
    auto oldEncoding = oldType.getEncoding().cast<BlockedEncodingAttr>();
    auto shape = oldType.getShape();
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
