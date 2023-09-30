#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include <iterator>
#include <numeric>

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct SimplifyConstantLoadsPattern : public OpRewritePattern<triton::LoadOp> {
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
  SimplifyConstantLoadsPattern(MLIRContext *context, ModuleAxisInfoAnalysis& axisInfoAnalysis) : OpRewritePattern<triton::LoadOp>(context, /*benefit=*/1), axisInfoAnalysis(axisInfoAnalysis) {}

  mlir::LogicalResult matchAndRewrite(triton::LoadOp loadOp,
                                     PatternRewriter &rewriter) const override {

    Value inputPtr = loadOp.getPtr();
    Value outputVal = loadOp.getResult();

    auto rttInputType = inputPtr.getType().dyn_cast<RankedTensorType>();
    auto rttOutputType = outputVal.getType().dyn_cast<RankedTensorType>();
    if (!rttInputType || !rttOutputType) {
      return failure();
    }

    auto inputEncoding = rttOutputType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
    if (!inputEncoding) {
      return failure();
    }

    auto elemsPerThread = triton::gpu::getElemsPerThread(rttInputType);
    auto shape = rttInputType.getShape();
    auto constancy = axisInfoAnalysis.getAxisInfo(inputPtr)->getConstancy();
    auto rank = constancy.size();

    if (elemsPerThread.size() != rank || shape.size() != rank) {
      // Shouldn't happen; number of dimensions should match
      return failure();
    }

    // Already a single load
    if (product<unsigned>(elemsPerThread) == 1) {
      return failure();
    }

    for (size_t i = 0; i < constancy.size(); ++i) {
      if (constancy[i] < elemsPerThread[i] || elemsPerThread[i] % constancy[i] != 0) {
        // This can be improved; but for now, we only apply this if the entire
        // load in the thread can be converted to a single load.
        return failure();
      }
    }

    // WIP - TODOs:
    // 1. Not sure if the part below builds yet...
    // 2. We also need to take slices of "mask", "other" https://github.com/openai/triton/blob/e0edb70f78a3702f727bbf1e9c2977d8f90bf530/include/triton/Dialect/Triton/IR/TritonOps.td#L136
    // 3. IDK, we may need to wrap the arguments to ExtractSliceOp in SmallVector<OpFoldResult> like https://github.com/openai/triton/blob/e0edb70f78a3702f727bbf1e9c2977d8f90bf530/lib/Dialect/TritonGPU/Transforms/Prefetch.cpp#L128
    // 4. Also pass all the other parameters for LoadOp
    // 5. Need to splat/broadcast the output

    // SmallVector<unsigned, 4> newShape;
    // for (size_t i = 0; i < shape.size(); ++i) {
    //   assert(shape[i] % elemsPerThread[i] == 0);
    //   newShape.push_back(shape[i] / elemsPerThread[i]);
    // }

    // SmallVector<unsigned> newSizePerThread(rank, 1);
    // SmallVector<unsigned> offsets(rank, 0);

    // auto newEncoding = triton::gpu::BlockedEncodingAttr::get(&getContext(), newShape, newSizePerThread, inputEncoding.getOrder(), triton::gpu::getNumWarpsPerCTA(inputEncoding), triton::gpu::getCTALayout(inputEncoding), triton::gpu::getCTALayout(inputEncoding));

    // rewriter.setInsertPoint(loadOp);
    // Value slice = rewriter.create<triton::gpu::ExtractSliceOp>(loadOp.getLoc(), RankedTensorType::(newShape, rttInputType.getElementType(), newEncoding), inputPtr,offsets, newShape, /*strides=*/elemsPerThread);
    // rewriter.create<triton::gpu::LoadOp>(loadOp.getLoc(), slice, loadOp.);

    return failure();
  }
};


struct SimplifyConstantLoads : public TritonGPUSimplifyConstantLoadsBase<SimplifyConstantLoads> {
  void runOnOperation() override {
    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<SimplifyConstantLoadsPattern>(context, axisInfoAnalysis);

    Region region(moduleOp);

    if (applyPatternsAndFoldGreedily(region, std::move(patterns)).failed()) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUSimplifyConstantLoadsPass() {
  return std::make_unique<SimplifyConstantLoads>();
}
