#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <numeric>

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

namespace {

struct LayoutInfo {
  ArrayRef<int64_t> shape;
  ArrayRef<unsigned> order;

  bool operator==(const LayoutInfo& other) const {
    return shape == other.shape && order == other.order;
  }
};

} // namespace

template <>
struct llvm::DenseMapInfo<LayoutInfo> {
  static LayoutInfo getEmptyKey() { return {{}, {}}; }
  static LayoutInfo getTombstoneKey() { return {{-1}, {unsigned(-1)}}; }
  static unsigned getHashValue(const LayoutInfo& val) {
    unsigned shape_hash = DenseMapInfo<ArrayRef<int64_t>>::getHashValue(val.shape);
    unsigned val_hash = DenseMapInfo<ArrayRef<int64_t>>::getHashValue(val.shape);
    return DenseMapInfo<std::pair<unsigned, unsigned>>::getHashValue({shape_hash, val_hash});
  }
  static bool isEqual(const LayoutInfo& lhs, const LayoutInfo& rhs) {
    return lhs == rhs;
  }
};

typedef DenseMap<Value, std::function<Type(Type)>> LayoutMap;
typedef DenseMap<Value, LayoutInfo> ValueToLayoutInfo;
typedef DenseMap<LayoutInfo, int64_t> LayoutInfoMap;

namespace {

class ReplaceLayouts : public mlir::RewritePattern {
  LayoutInfoMap& layoutInfoMap;
  ValueToLayoutInfo& valueToLayoutInfo;
  ReplaceLayouts(mlir::MLIRContext *context, LayoutInfoMap& layoutInfoMap, ValueToLayoutInfo& valueToLayoutInfo) : mlir::RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/0, context), layoutInfoMap(layoutInfoMap), valueToLayoutInfo(valueToLayoutInfo) {
  }

  LogicalResult matchAndRewrite(Operation *operation, PatternRewriter& rewriter) const override {
    Value ptr;
    return failure();
  }
};

} // namespace

struct CoalescePass : public TritonGPUCoalesceBase<CoalescePass> {

  LayoutInfo getLayoutInfo(ModuleAxisInfoAnalysis &axisInfoAnalysis,
                           Value ptr) {
    auto origType = ptr.getType().cast<RankedTensorType>();
    // Get the contiguity order of `ptr`
    auto order = argSort(axisInfoAnalysis.getAxisInfo(ptr)->getContiguity());
    return {origType.getShape(), order};
  }

  unsigned aggregateLayoutInfo(const LayoutInfo& layoutInfo, ModuleAxisInfoAnalysis& axisInfoAnalysis, Value ptr, int numWarps, int threadsPerWarp) {
    auto origType = ptr.getType().cast<RankedTensorType>();
    auto& order = layoutInfo.order;

    int numElems = product(origType.getShape());
    int numThreads = numWarps * threadsPerWarp;
    int numElemsPerThread = std::max(numElems / numThreads, 1);
    // Thread tile size depends on memory alignment
    unsigned elemNumBits = triton::getPointeeBitWidth(origType);
    unsigned elemNumBytes = std::max(elemNumBits / 8, 1u);

    unsigned maxMultipleBytes =
        axisInfoAnalysis.getAxisInfo(ptr)->getDivisibility(order[0]);
    unsigned maxMultiple = std::max(maxMultipleBytes / elemNumBytes, 1u);
    unsigned maxContig =
        axisInfoAnalysis.getAxisInfo(ptr)->getContiguity(order[0]);
    unsigned alignment = std::min(maxMultiple, maxContig);
    unsigned perThread = std::min(alignment, 128 / elemNumBits);
    perThread = std::min<unsigned>(perThread, numElemsPerThread);

    return perThread;
  }

  // TODO: refactor this to:
  //  1. get the layout information
  //  2. get the max grouped prediction
  Attribute getCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 Value ptr, int numWarps, int threadsPerWarp) {
    auto origType = ptr.getType().cast<RankedTensorType>();
    // Get the shape of the tensor.
    size_t rank = origType.getRank();
    // Get the contiguity order of `ptr`
    auto order = argSort(axisInfoAnalysis.getAxisInfo(ptr)->getContiguity());
    // The desired divisibility is the maximum divisibility
    // among all dependent pointers who have the same order as
    // `ptr`
    SetVector<Value> withSameOrder;
    withSameOrder.insert(ptr);
    if (ptr.getDefiningOp())
      for (Operation *op : mlir::multiRootGetSlice(ptr.getDefiningOp())) {
        for (Value val : op->getResults()) {
          if (val.getType() != origType)
            continue;
          auto currOrder =
              argSort(axisInfoAnalysis.getAxisInfo(val)->getContiguity());
          if (order == currOrder)
            withSameOrder.insert(val);
        }
      }
    int numElems = product(origType.getShape());
    int numThreads = numWarps * threadsPerWarp;
    int numElemsPerThread = std::max(numElems / numThreads, 1);
    // Thread tile size depends on memory alignment
    SmallVector<unsigned, 4> sizePerThread(rank, 1);
    unsigned elemNumBits = triton::getPointeeBitWidth(origType);
    unsigned elemNumBytes = std::max(elemNumBits / 8, 1u);
    unsigned perThread = 1;
    for (Value val : withSameOrder) {
      unsigned maxMultipleBytes =
          axisInfoAnalysis.getAxisInfo(val)->getDivisibility(order[0]);
      unsigned maxMultiple = std::max(maxMultipleBytes / elemNumBytes, 1u);
      unsigned maxContig =
          axisInfoAnalysis.getAxisInfo(val)->getContiguity(order[0]);
      unsigned alignment = std::min(maxMultiple, maxContig);
      unsigned currPerThread = std::min(alignment, 128 / elemNumBits);
      perThread = std::max(perThread, currPerThread);
    }
    sizePerThread[order[0]] = std::min<int>(perThread, numElemsPerThread);
    SmallVector<unsigned> dims(rank);
    std::iota(dims.begin(), dims.end(), 0);
    // create encoding
    Attribute encoding = triton::gpu::BlockedEncodingAttr::get(
        &getContext(), origType.getShape(), sizePerThread, order, numWarps,
        threadsPerWarp);
    return encoding;
  }

  std::function<Type(Type)>
  getTypeConverter(ModuleAxisInfoAnalysis &axisInfoAnalysis, Value ptr,
                   int numWarps, int threadsPerWarp) {
    Attribute encoding =
        getCoalescedEncoding(axisInfoAnalysis, ptr, numWarps, threadsPerWarp);
    return [encoding](Type _type) {
      RankedTensorType type = _type.cast<RankedTensorType>();
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   encoding);
    };
  }

/*
  void replaceOp(ModuleAxisInfoAnalysis &axisInfoAnalysis, const LayoutInfo &layoutInfo, const LayoutInfoMap& layoutInfoMap, Operation *op, Value ptr,
                  OpBuilder builder, int numWarps, int threadsPerWarp) {
    RankedTensorType ty = ptr.getType().template dyn_cast<RankedTensorType>();
    if (!ty)
      return;

    auto origShape = ty.getShape();
    auto origRank = ty.getRank();
    SmallVector<unsigned, 4> sizePerThread(rank, 1);
    sizePerThread[order[0]] = layoutInfoMap[layoutInfo];
    Attribute encoding = triton::gpu::BlockedEncodingAttr::get(
      &getContext(), origType.getShape(), sizePerThread, layoutInfo.order, numWarps,
      threadsPerWarp);

    auto convertType = [&](Type _type) {
      RankedTensorType = _type.cast<RankedTensorType>();
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                  encoding);
    };

    SmallVector<Value, 4> newArgs;
    IRMapping
    for (auto v : op->getOperands()) {
      auto vTyp = v.getType().dyn_cast<RankedTensorType>();
      if (vTy && !vTy.getEncoding().isa<triton::gpu::SharedEncodingAttr>())
        newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), convertType(v.getType()), v
        ));
      else
        newArgs.push_back(v);
    }

    Operation *newOp = builder.clone(op);
    newOp


  }
*/

  template <class T>
  void coalesceOp(LayoutMap &layoutMap, Operation *op, Value ptr,
                  OpBuilder builder) {
    RankedTensorType ty = ptr.getType().template dyn_cast<RankedTensorType>();
    if (!ty)
      return;
    auto convertType = layoutMap.lookup(ptr);
    // convert operands
    SmallVector<Value, 4> newArgs;
    for (auto v : op->getOperands()) {
      auto vTy = v.getType().dyn_cast<RankedTensorType>();
      if (vTy && !vTy.getEncoding().isa<triton::gpu::SharedEncodingAttr>())
        newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), convertType(v.getType()), v));
      else
        newArgs.push_back(v);
    }
    // convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      bool is_async = std::is_same<T, triton::gpu::InsertSliceAsyncOp>::value;
      newTypes.push_back(is_async ? t : convertType(t));
    }
    // construct new op with the new encoding
    Operation *newOp =
        builder.create<T>(op->getLoc(), newTypes, newArgs, op->getAttrs());
    // cast the results back to the original layout
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

  void runOnOperation() override {
    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    ValueToLayoutInfo perOpLayoutInfo;
    moduleOp.walk([&](Operation *curr) {
      Value ptr = curr->getResult();
      perOpLayoutInfo[ptr] = getLayoutInfo(axisInfoAnalysis, ptr);
    });

    LayoutInfoMap layoutInfoMap;
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
      auto mod = curr->getParentOfType<ModuleOp>();
      int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
      int threadsPerWarp =
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      const auto& layout = perOpLayoutInfo[ptr];
      unsigned elementsPerThread = aggregateLayoutInfo(layout, axisInfoAnalysis, ptr, numWarps, threadsPerWarp);
      if (!layoutInfoMap.contains(layout)) {
        layoutInfoMap[layout] = 1;
      }
      layoutInfoMap[layout] = std::max<unsigned>(layoutInfoMap[layout], elementsPerThread);
    });

    // For each i/o operation, we determine what layout
    // the pointers should have for best memory coalescing
    LayoutMap layoutMap;
    // LayoutInfoMap layoutInfoMap;
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
      RankedTensorType ty = ptr.getType().template dyn_cast<RankedTensorType>();
      if (!ty || !ty.getElementType().isa<PointerType>())
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
    /*
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
    */
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUCoalescePass() {
  return std::make_unique<CoalescePass>();
}
