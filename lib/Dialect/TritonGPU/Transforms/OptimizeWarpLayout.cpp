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
      
    }

    // TODO: Implement warp layout optimization
  }

private:
  DenseMap<Value, Attribute> layouts;
};

} // namespace gpu
} // namespace triton
} // namespace mlir
