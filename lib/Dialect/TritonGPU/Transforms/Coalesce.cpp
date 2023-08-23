#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include <iterator>
#include <numeric>

#include <iostream>

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

template <class T> SmallVector<unsigned, 4> argSort(const T &arr) {
  SmallVector<unsigned, 4> ret(arr.size());
  std::iota(ret.begin(), ret.end(), 0);
  std::stable_sort(ret.begin(), ret.end(),
                   [&](unsigned x, unsigned y) { return arr[x] > arr[y]; });
  return ret;
}

typedef DenseMap<Value, std::pair<std::function<Type(Type)>, std::function<bool(Type)>>> LayoutMap;

// TODO better name
RankedTensorType maximizeContiguousShape(
  RankedTensorType origType,
  MLIRContext* context,
  unsigned numWarps,
  unsigned threadsPerWarp
) {
  ArrayRef<int64_t> shape = origType.getShape();
  int rank = shape.size();

  SmallVector<unsigned> order(rank);
  // TODO handle order correctly
  std::iota(order.rbegin(), order.rend(), 0);
  SmallVector<unsigned> sizePerThread(rank, 1);
  auto shapePerCTA = triton::gpu::getShapePerCTA(origType);

  // TODO: handle multi-dimensional cases correctly here
  while (sizePerThread[rank-1] * numWarps * threadsPerWarp < shapePerCTA[rank-1]) {
    sizePerThread[rank-1] *= 2;
  }

  auto CTALayout = triton::gpu::getCTALayout(origType.getEncoding());
  auto newEncoding = triton::gpu::BlockedEncodingAttr::get(
    context, shape, sizePerThread, order, numWarps, threadsPerWarp, CTALayout);
  auto newType = RankedTensorType::get(shape, origType.getElementType(), newEncoding);
  return newType;
}

class MaximizeCoalesceOp {
public:
  virtual LogicalResult matchAndRewrite(Operation* op, PatternRewriter& rewriter, MLIRContext* context, unsigned numWarps, unsigned threadsPerWarp) = 0;
  virtual bool matchOp(Operation* op) = 0;
};

template <typename OpTy>
class MaximizeCoalesceOpImpl : public MaximizeCoalesceOp {
public:
  // TODO(dberard): split matchAndRewrite to match / rewrite.
  //   Benefit: we can use it as part of the initial "can we rewrite?" check...
  LogicalResult matchAndRewrite(Operation* op, PatternRewriter& rewriter, MLIRContext* context, unsigned numWarps, unsigned threadsPerWarp) override final {
    return matchAndRewriteImpl(cast<OpTy>(op), rewriter, context, numWarps, threadsPerWarp);
  }
  bool matchOp(Operation* op) override final {
    return isa<OpTy>(op);
  }
  virtual LogicalResult matchAndRewriteImpl(OpTy op, PatternRewriter& rewriter, MLIRContext* context, unsigned numWarps, unsigned threadsPerWarp) = 0;
};

class MaximizeCoalesceConstantOp : public MaximizeCoalesceOpImpl<arith::ConstantOp> {
  LogicalResult matchAndRewriteImpl(arith::ConstantOp op, PatternRewriter& rewriter, MLIRContext* context, unsigned numWarps, unsigned threadsPerWarp) override final {
    // TODO(dberard): unify origType / oldType naming scheme...
    auto origType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!origType) {
      return failure();
    }
    auto origEncoding = origType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
    if (!origEncoding) {
      return failure();
    }
    auto value = op.getValue().dyn_cast<DenseElementsAttr>();
    if (!value) {
      return failure();
    }
    RankedTensorType newType = maximizeContiguousShape(
      origType, context, numWarps, threadsPerWarp
    );
    if (newType == origType) {
      return failure();
    }
    auto getSizePerThread = [](RankedTensorType ty) -> unsigned {
      return ty
        .getEncoding()
        .cast<triton::gpu::BlockedEncodingAttr>()
        .getSizePerThread()[0];
    };
    std::cerr << " Replacing " << op << " , " << getSizePerThread(origType) << " with newop , " << getSizePerThread(newType) << std::endl;
    std::cerr << "     value type " << getSizePerThread(value.getType().cast<RankedTensorType>()) << std::endl;
    auto newShapedType = newType.cast<ShapedType>();
    value = value.reshape(newShapedType);
    auto newOp = rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newShapedType, value);
    return success();
  }
};

class MaximizeCoalescePattern : public RewritePattern {
public:
  explicit MaximizeCoalescePattern(MLIRContext* context, unsigned numWarps, unsigned threadsPerWarp) : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context), numWarps(numWarps), threadsPerWarp(threadsPerWarp) {
    visitors.push_back(std::make_unique<MaximizeCoalesceConstantOp>());
  }

  LogicalResult matchAndRewrite(Operation* op, PatternRewriter& rewriter) const override {
    std::cerr << "Handling " << op << std::endl;
    for (auto& visitor : visitors) {
      if (visitor->matchOp(op)) {
        std::cerr << "  op is handled specially!" << std::endl;
        return visitor->matchAndRewrite(op, rewriter, getContext(), numWarps, threadsPerWarp);
      }
    }
    if (op->getResults().size() != 1) {
      return failure();
    }
    const auto& oldResult = op->getResults()[0];
    auto rttType = oldResult.getType().dyn_cast<RankedTensorType>();
    if (!rttType) {
      return failure();
    }
    auto encoding = rttType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
    if (!encoding) {
      return failure();
    }

    auto newType = maximizeContiguousShape(rttType, getContext(), numWarps, threadsPerWarp);

    SmallVector<Type, 1> newReturnTypes;
    newReturnTypes.push_back(std::move(newType));

    // TODO: is this all the state we need?
    OperationState newState(op->getLoc(), op->getName());
    newState.addOperands(op->getOperands());
    newState.addTypes(std::move(newReturnTypes));
    newState.addAttributes(op->getAttrs());

    if (newType == rttType) {
      return failure();
    }

    Operation* newOp = rewriter.create(newState);
    auto getSizePerThread = [](Operation* oper) -> unsigned {
      return oper
        ->getResults()[0]
        .getType()
        .cast<RankedTensorType>()
        .getEncoding()
        .cast<triton::gpu::BlockedEncodingAttr>()
        .getSizePerThread()[0];
    };
    std::cerr << " Replacing " << op << " , " << getSizePerThread(op) << " with " << newOp << " , " << getSizePerThread(newOp) << std::endl;
    rewriter.replaceOp(op, newOp->getResults());

    return success();
  }
private:
  unsigned numWarps;
  unsigned threadsPerWarp;
  std::vector<std::unique_ptr<MaximizeCoalesceOp>> visitors; // TODO rename
};

struct CoalescePass : public TritonGPUCoalesceBase<CoalescePass> {
  Attribute getCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 Value ptr, int numWarps, int threadsPerWarp) {
    auto refType = ptr.getType();
    if (refType.isa<PointerType>())
      refType = refType.cast<PointerType>().getPointeeType();
    auto refTensorType = refType.cast<RankedTensorType>();

    // TODO(Keren): integrate it into AxisInfoAnalysis
    // Get axis info
    auto queryAxisInfo = [&](const Value &val) -> AxisInfo {
      auto valType = val.getType();
      // Tensor pointer
      // TODO(Chenggang): encoding for tensor pointers is meaningless, remove
      // these later while merging into the GitHub main
      if (auto ptrType = valType.dyn_cast<PointerType>()) {
        auto tensorTy = ptrType.getPointeeType().dyn_cast<RankedTensorType>();
        assert(tensorTy);
        auto makeTensorPtr = getMakeTensorPtrOp(val);
        auto order = makeTensorPtr.getOrder();
        auto tileShape = triton::gpu::getShapePerCTA(tensorTy);
        size_t rank = order.size();
        auto elemSizeInBytes =
            tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
        SmallVector<int64_t> contiguity(rank, 1);
        SmallVector<int64_t> divisibility(rank, 1);
        SmallVector<int64_t> constancy(rank, 1);
        // The contiguity in `order[0]` is `tileShape[order[0]]`
        // The divisibility in `order[0]` is 16
        // TODO[goostavz]: confirm the legality of it
        contiguity[order[0]] = tileShape[order[0]];
        divisibility[order[0]] = 16 * 8 / elemSizeInBytes;
        return AxisInfo(contiguity, divisibility, constancy);
      }
      // Normal cases
      assert(valType.isa<RankedTensorType>());
      return *axisInfoAnalysis.getAxisInfo(val);
    };

    // Get the contiguity order of `ptr`
    SmallVector<unsigned> order;
    if (auto ptrType = ptr.getType().dyn_cast<PointerType>()) {
      // Tensor pointer
      auto makeTensorPtr = getMakeTensorPtrOp(ptr);
      std::copy(makeTensorPtr.getOrder().begin(),
                makeTensorPtr.getOrder().end(), std::back_inserter(order));
    } else {
      // Normal cases
      order = argSort(queryAxisInfo(ptr).getContiguity());
    }

    // The desired divisibility is the maximum divisibility
    // among all dependent pointers who have the same order as
    // `ptr`.
    // We only do it for normal tensors of pointers, not tensor pointers.
    SetVector<Value> withSameOrder;
    withSameOrder.insert(ptr);
    if (refType.isa<RankedTensorType>() && ptr.getDefiningOp()) {
      for (Operation *op : mlir::multiRootGetSlice(ptr.getDefiningOp())) {
        for (Value val : op->getResults()) {
          if (val.getType() != refTensorType)
            continue;
          auto currOrder =
              argSort(axisInfoAnalysis.getAxisInfo(val)->getContiguity());
          if (order == currOrder)
            withSameOrder.insert(val);
        }
      }
    }

    auto shapePerCTA = triton::gpu::getShapePerCTA(refTensorType);
    int numElems = product<int64_t>(shapePerCTA);
    int numThreads = numWarps * threadsPerWarp;
    int numElemsPerThread = std::max(numElems / numThreads, 1);

    // For tensor of pointers, the element to access is the pointee type;
    // while for tensor pointer type (`refType` is directly the final shape),
    // the element to access is itself.
    auto typeForMem = refTensorType.getElementType().isa<PointerType>()
                          ? refTensorType.getElementType()
                                .cast<PointerType>()
                                .getPointeeType()
                          : refTensorType.getElementType();

    // Thread tile size depends on memory alignment
    SmallVector<unsigned, 4> sizePerThread(refTensorType.getRank(), 1);
    unsigned elemNumBits = typeForMem.getIntOrFloatBitWidth();
    unsigned elemNumBytes = std::max(elemNumBits / 8, 1u);
    unsigned perThread = 1;
    for (Value val : withSameOrder) {
      auto valInfo = queryAxisInfo(val);
      unsigned maxMultipleBytes = valInfo.getDivisibility(order[0]);
      unsigned maxMultiple = std::max(maxMultipleBytes / elemNumBytes, 1u);
      unsigned maxContig =
          std::min(valInfo.getContiguity(order[0]), shapePerCTA[order[0]]);
      unsigned alignment = std::min(maxMultiple, maxContig);
      unsigned currPerThread = std::min(alignment, 128 / elemNumBits);
      perThread = std::max(perThread, currPerThread);
    }
    sizePerThread[order[0]] = std::min<int>(perThread, numElemsPerThread);

    auto CTALayout = triton::gpu::getCTALayout(refTensorType.getEncoding());
    return triton::gpu::BlockedEncodingAttr::get(
        &getContext(), refTensorType.getShape(), sizePerThread, order, numWarps,
        threadsPerWarp, CTALayout);
  }

  std::pair<std::function<Type(Type)>, std::function<bool(Type)>>
  getTypeConverter(ModuleAxisInfoAnalysis &axisInfoAnalysis, Value ptr,
                   int numWarps, int threadsPerWarp) {
    Attribute encoding =
        getCoalescedEncoding(axisInfoAnalysis, ptr, numWarps, threadsPerWarp);
    return std::make_pair(
      [encoding](Type type) {
        RankedTensorType tensorType = type.cast<RankedTensorType>();
        return RankedTensorType::get(tensorType.getShape(),
                                    tensorType.getElementType(), encoding);
      },
      [encoding](Type type) {
        RankedTensorType tensorType = type.cast<RankedTensorType>();
        const auto& newEncoding = encoding.cast<triton::gpu::BlockedEncodingAttr>();
        const auto& newOrder = newEncoding.getOrder();
        const auto& newSizePerThread = newEncoding.getSizePerThread();
        const auto& origEncoding = tensorType.getEncoding().cast<triton::gpu::BlockedEncodingAttr>();
        const auto& origOrder = origEncoding.getOrder();
        const auto& origSizePerThread = origEncoding.getSizePerThread();
        // TODO: I feel like there are more edge cases here...
        if (newSizePerThread[newOrder[0]] > origSizePerThread[origOrder[0]]) {
          return true;
        }
        return false;
      }
    );
  }

  template <class T>
  void coalesceOp(LayoutMap &layoutMap, Operation *op, Value ptr,
                  OpBuilder builder) {
    if (!layoutMap.count(ptr))
      return;

    // Convert operands
    // For load/store with tensor pointers, we don't have to change the
    // operands' type, we do this by changing the outputs' type of
    // `make_tensor_ptr`
    auto [convertType, shouldConvertType] = layoutMap.lookup(ptr);
    SmallVector<Value, 4> newArgs;
    bool shouldConvert = false;

    // Convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      bool isAsync = std::is_same<T, triton::gpu::InsertSliceAsyncOp>::value;
      newTypes.push_back(isAsync ? t : convertType(t));
      if (isAsync) {
        shouldConvert |= shouldConvertType(t);
      }
    }

    for (auto operand : op->getOperands()) {
      auto tensorType = operand.getType().dyn_cast<RankedTensorType>();
      if (tensorType &&
          !tensorType.getEncoding().isa<triton::gpu::SharedEncodingAttr>()) {
        newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), convertType(tensorType), operand));
        shouldConvert |= shouldConvertType(tensorType);
      }
      else
        newArgs.push_back(operand);
    }

    if (!shouldConvert) {
      return;
    }

    // Construct new op with the new encoding
    Operation *newOp =
        builder.create<T>(op->getLoc(), newTypes, newArgs, op->getAttrs());

    // Cast the results back to the original layout
    for (size_t i = 0; i < op->getNumResults(); i++) {
      Value newResult = newOp->getResult(i);
      if (newTypes[i] != op->getResultTypes()[i]) {
        newResult = builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), op->getResult(i).getType(), newResult);
      }
      op->getResult(i).replaceAllUsesWith(newResult);
    }
    op->erase();
  }

  void coalesceMakeTensorPtrOpResult(LayoutMap &layoutMap, Operation *op,
                                     Value ptr, OpBuilder builder) {
    if (!layoutMap.count(ptr))
      return;

    // Convert result type
    auto [convertType, shouldConvertType] = layoutMap.lookup(ptr);
    auto ptrType = ptr.getType().cast<PointerType>();
    auto resultTensorType = convertType(ptrType.getPointeeType());
    auto newResultType =
        PointerType::get(resultTensorType, ptrType.getAddressSpace());

    // Build new operation and replace
    Operation *newOp = builder.create<MakeTensorPtrOp>(
        op->getLoc(), newResultType, op->getOperands(), op->getAttrs());
    op->getResult(0).replaceAllUsesWith(newOp->getResult(0));
    op->erase();
  }

  void runOnOperation() override {
    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    std::cerr << " runOnOperation: Coalesce, want to apply MaximizeCoalescePattern" << std::endl;

    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(moduleOp);
    int threadsPerWarp =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(moduleOp);

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<MaximizeCoalescePattern>(context, numWarps, threadsPerWarp);

    if (applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)).failed()) {
      std::cerr << " applyPatternsAndFoldGreedily failed" << std::endl;
      return signalPassFailure();
    }
    moduleOp.dump();

    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // For each i/o operation, we determine what layout
    // the pointers should have for best memory coalescing
    LayoutMap layoutMap;
    moduleOp.walk([&](Operation *curr) {
      Value ptr;
      if (auto op = dyn_cast<triton::LoadOp>(curr))
        ptr = op.getPtr();
      if (auto op = dyn_cast<triton::AtomicRMWOp>(curr))
        ptr = op.getPtr();
      if (auto op = dyn_cast<triton::AtomicCASOp>(curr))
        ptr = op.getPtr();
      if (auto op = dyn_cast<triton::gpu::InsertSliceAsyncOp>(curr))
        ptr = op.getSrc();
      if (auto op = dyn_cast<triton::StoreOp>(curr))
        ptr = op.getPtr();
      if (!ptr)
        return;
      // We only convert `tensor<tt.ptr<>>` or `tt.ptr<tensor<>>` load/store
      bool isPtrTensor = false, isTensorPointer = false;
      if (auto tensorType = ptr.getType().dyn_cast<RankedTensorType>())
        isPtrTensor = tensorType.getElementType().isa<PointerType>();
      if (auto ptrType = ptr.getType().dyn_cast<PointerType>())
        isTensorPointer = ptrType.getPointeeType().isa<RankedTensorType>();
      if (!isPtrTensor && !isTensorPointer)
        return;
      auto mod = curr->getParentOfType<ModuleOp>();
      int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
      int threadsPerWarp =
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      auto convertType =
          getTypeConverter(axisInfoAnalysis, ptr, numWarps, threadsPerWarp);
      layoutMap[ptr] = convertType;
    });

    // For each memory op that has a layout L1:
    // 1. Create a coalesced memory layout L2 of the pointer operands
    // 2. Convert all operands from layout L1 to layout L2
    // 3. Create a new memory op that consumes these operands and
    //    produces a tensor with layout L2
    // 4. Convert the output of this new memory op back to L1
    // 5. Replace all the uses of the original memory op by the new one
    moduleOp.walk([&](Operation *curr) {
      OpBuilder builder(curr);
      if (auto load = dyn_cast<triton::LoadOp>(curr)) {
        coalesceOp<triton::LoadOp>(layoutMap, curr, load.getPtr(), builder);
        return;
      }
      if (auto op = dyn_cast<triton::AtomicRMWOp>(curr)) {
        coalesceOp<triton::AtomicRMWOp>(layoutMap, curr, op.getPtr(), builder);
        return;
      }
      if (auto op = dyn_cast<triton::AtomicCASOp>(curr)) {
        coalesceOp<triton::AtomicCASOp>(layoutMap, curr, op.getPtr(), builder);
        return;
      }
      if (auto load = dyn_cast<triton::gpu::InsertSliceAsyncOp>(curr)) {
        coalesceOp<triton::gpu::InsertSliceAsyncOp>(layoutMap, curr,
                                                    load.getSrc(), builder);
        return;
      }
      if (auto store = dyn_cast<triton::StoreOp>(curr)) {
        coalesceOp<triton::StoreOp>(layoutMap, curr, store.getPtr(), builder);
        return;
      }
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUCoalescePass() {
  return std::make_unique<CoalescePass>();
}
