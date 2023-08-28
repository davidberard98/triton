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

// In order to coalesce memory accesses, we might require a certain ordering
// and a minimum sizePerThread along the inner dimension. This struct stores
// those two properties.
struct CoalescedConstraint {
  SmallVector<unsigned> order;
  int64_t sizePerThread; // for sizePerThread[order[0]]
};

typedef DenseMap<Value, std::optional<CoalescedConstraint>> LayoutMap;

// TODO remove dump() and cerr/cout.
// TODO: calculate this once instead of for each conversion...
RankedTensorType maximizeContiguousShape(RankedTensorType origType,
                                         MLIRContext *context,
                                         const CoalescedConstraint &constraint,
                                         unsigned numWarps,
                                         unsigned threadsPerWarp) {
  ArrayRef<int64_t> shape = origType.getShape();
  int rank = shape.size();

  SmallVector<unsigned, 4> sizePerThread(rank, 1);
  auto shapePerCTA = triton::gpu::getShapePerCTA(origType);

  sizePerThread[constraint.order[0]] = constraint.sizePerThread;

  auto CTALayout = triton::gpu::getCTALayout(origType.getEncoding());
  auto newEncoding = triton::gpu::BlockedEncodingAttr::get(
      context, shape, sizePerThread, constraint.order, numWarps, threadsPerWarp,
      CTALayout);
  auto newType =
      RankedTensorType::get(shape, origType.getElementType(), newEncoding);
  return newType;
}

class ReassignEncodingSupportedState {
public:
  void addShape(ArrayRef<int64_t> shape) { shapes.insert(std::move(shape)); }
  int uniqueShapes() const { return shapes.size(); }
  void markUnsupported() { supported = false; }
  bool isSupported() const { return supported; }

private:
  SetVector<ArrayRef<int64_t>> shapes;
  bool supported{true};
};

class ReassignEncodingOpHandler {
public:
  ReassignEncodingOpHandler() = default;
  virtual ~ReassignEncodingOpHandler() = default;
  virtual LogicalResult
  matchAndRewrite(Operation *op, PatternRewriter &rewriter,
                  MLIRContext *context, const CoalescedConstraint &constraint,
                  unsigned numWarps, unsigned threadsPerWarp) const = 0;
  virtual bool matchOp(Operation *op) const = 0;
  virtual bool supported(Operation *op,
                         ReassignEncodingSupportedState &state) const = 0;
};

template <typename OpTy>
class ReassignEncodingOpHandlerImpl : public ReassignEncodingOpHandler {
public:
  ReassignEncodingOpHandlerImpl() = default;
  virtual ~ReassignEncodingOpHandlerImpl() = default;
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter,
                                MLIRContext *context,
                                const CoalescedConstraint &constraint,
                                unsigned numWarps,
                                unsigned threadsPerWarp) const override {
    return matchAndRewriteImpl(cast<OpTy>(op), rewriter, context, constraint,
                               numWarps, threadsPerWarp);
  }
  virtual LogicalResult
  matchAndRewriteImpl(OpTy op, PatternRewriter &rewriter, MLIRContext *context,
                      const CoalescedConstraint &constraint, unsigned numWarps,
                      unsigned threadsPerWarp) const = 0;
  bool matchOp(Operation *op) const override final { return isa<OpTy>(op); }
  bool supported(Operation *op,
                 ReassignEncodingSupportedState &state) const override {
    for (const Value &result : op->getResults()) {
      if (result.getType().template dyn_cast<PointerType>()) {
        return false; // block pointer not supported
      }
      if (!result.getType().template dyn_cast<TensorType>()) {
        continue; // ignore scalar types
      }
      auto type = result.getType().template dyn_cast<RankedTensorType>();
      if (!type) {
        std::cerr << "   fails: not RankedTensorType" << std::endl;
        return false;
      }
      auto encoding =
          type.getEncoding()
              .template dyn_cast<triton::gpu::BlockedEncodingAttr>();
      if (!encoding) {
        std::cerr << "   fails: not BlockedEncodingAttr:"
                  << op->getName().getStringRef().str() << std::endl;
        return false;
      }
      // For simplicity, only support graphs where all tensors have the same
      // shape
      state.addShape(type.getShape());
    }
    return supportedImpl(cast<OpTy>(op));
  }
  virtual bool supportedImpl(OpTy op) const = 0;
};

class ReassignEncodingConstantOp
    : public ReassignEncodingOpHandlerImpl<arith::ConstantOp> {
  LogicalResult
  matchAndRewriteImpl(arith::ConstantOp op, PatternRewriter &rewriter,
                      MLIRContext *context,
                      const CoalescedConstraint &constraint, unsigned numWarps,
                      unsigned threadsPerWarp) const override final {
    auto origType = op.getResult().getType().dyn_cast<RankedTensorType>();
    if (!origType) {
      return failure();
    }
    auto origEncoding =
        origType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
    if (!origEncoding) {
      return failure();
    }
    auto value = op.getValue().dyn_cast<DenseElementsAttr>();
    if (!value) {
      return failure();
    }
    RankedTensorType newType = maximizeContiguousShape(
        origType, context, constraint, numWarps, threadsPerWarp);
    if (newType == origType) {
      return failure();
    }
    auto getSizePerThread = [](RankedTensorType ty) -> unsigned {
      return ty.getEncoding()
          .cast<triton::gpu::BlockedEncodingAttr>()
          .getSizePerThread()[0];
    };
    std::cerr << " Replacing " << op << " , " << getSizePerThread(origType)
              << " with newop , " << getSizePerThread(newType) << std::endl;
    std::cerr << "     value type "
              << getSizePerThread(value.getType().cast<RankedTensorType>())
              << std::endl;
    auto newShapedType = newType.cast<ShapedType>();
    value = value.reshape(newShapedType);
    auto newOp = rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, newShapedType, value);
    return success();
  }

  bool supportedImpl(arith::ConstantOp op) const override final { return true; }
};

class ReassignEncodingGeneric : public ReassignEncodingOpHandler {
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter,
                                MLIRContext *context,
                                const CoalescedConstraint &constraint,
                                unsigned numWarps,
                                unsigned threadsPerWarp) const override final {
    if (op->getResults().size() == 0) {
      return failure();
    }
    assert(op->getResults().size() == 1 &&
           "Expected only 1 result from this op");
    const auto &oldResult = op->getResults()[0];
    auto rttType = oldResult.getType().dyn_cast<RankedTensorType>();
    if (!rttType) {
      return failure();
    }
    auto encoding =
        rttType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
    if (!encoding) {
      return failure();
    }

    auto newType = maximizeContiguousShape(rttType, context, constraint,
                                           numWarps, threadsPerWarp);

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

    Operation *newOp = rewriter.create(newState);
    auto getSizePerThread = [](Operation *oper) -> unsigned {
      return oper->getResults()[0]
          .getType()
          .cast<RankedTensorType>()
          .getEncoding()
          .cast<triton::gpu::BlockedEncodingAttr>()
          .getSizePerThread()[0];
    };
    std::cerr << " Replacing " << op << " , " << getSizePerThread(op)
              << " with " << newOp << " , " << getSizePerThread(newOp)
              << std::endl;
    rewriter.replaceOp(op, newOp->getResults());

    return success();
  }
  bool matchOp(Operation *op) const override {
    llvm_unreachable("matchOp not supported on ReassignEncodingGeneric");
  }
  bool supported(Operation *op,
                 ReassignEncodingSupportedState &state) const override final {
    return (op->getResults().size() <= 1);
  }
};

template <typename OpTy>
class ReassignEncodingSupported : public ReassignEncodingGeneric {
  bool matchOp(Operation *op) const override final { return isa<OpTy>(op); }
};

class ReassignEncodingArithOps : public ReassignEncodingGeneric {
  bool matchOp(Operation *op) const override final {
    // ConstantOp needs special handling for its value.
    return op->getName().getDialect()->getNamespace() == "arith" &&
           !isa<arith::ConstantOp>(op);
  }
};

class ReassignEncodingMathOps : public ReassignEncodingGeneric {
  bool matchOp(Operation *op) const override final {
    // ConstantOp needs special handling for its value.
    return op->getName().getDialect()->getNamespace() == "math" &&
           op->hasTrait<OpTrait::SameOperandsAndResultType>();
  }
};

class ReassignEncodingSameOpResultEncoding : public ReassignEncodingGeneric {
  bool matchOp(Operation *op) const override final {
    return op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() &&
           op->hasTrait<OpTrait::SameOperandsAndResultShape>();
  }
};

template <typename OpTy>
class ReassignEncodingUnsupported : public ReassignEncodingOpHandlerImpl<OpTy> {
  LogicalResult matchAndRewriteImpl(OpTy, PatternRewriter &, MLIRContext *,
                                    const CoalescedConstraint &, unsigned,
                                    unsigned) const override final {
    return failure();
  }

  bool supportedImpl(OpTy op) const override final { return false; }
};

class ReassignEncodingOpHandlerList {
public:
  template <typename T> void append() {
    // TODO rename
    visitors.push_back(std::make_unique<T>());
  }

  bool supported(Operation *op, ReassignEncodingSupportedState &state) const {
    for (const auto &visitor : visitors) {
      if (visitor->matchOp(op)) {
        return visitor->supported(op, state);
      }
    }
    // TODO make it default to just "false"
    // llvm_unreachable(("No handler for " +
    // op->getName().getStringRef()).str().c_str());
    std::cerr << " ! Unsupported op " << op->getName().getStringRef().str()
              << std::endl;
    return false;
  }

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter,
                                MLIRContext *context,
                                const CoalescedConstraint &constraint,
                                unsigned numWarps,
                                unsigned threadsPerWarp) const {
    for (const auto &visitor : visitors) {
      if (visitor->matchOp(op)) {
        return visitor->matchAndRewrite(op, rewriter, context, constraint,
                                        numWarps, threadsPerWarp);
      }
    }
    llvm_unreachable(
        ("No matchAndRewrite handler for " + op->getName().getStringRef())
            .str()
            .c_str());
  }

private:
  std::vector<std::unique_ptr<ReassignEncodingOpHandler>> visitors;
};

class ReassignEncodingPattern : public RewritePattern {
public:
  explicit ReassignEncodingPattern(MLIRContext *context,
                                   ReassignEncodingOpHandlerList handlers,
                                   const CoalescedConstraint &constraint,
                                   unsigned numWarps, unsigned threadsPerWarp)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        handlers(std::move(handlers)), constraint(std::move(constraint)),
        numWarps(numWarps), threadsPerWarp(threadsPerWarp) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    std::cerr << "Handling " << op << std::endl;
    return handlers.matchAndRewrite(op, rewriter, getContext(), constraint,
                                    numWarps, threadsPerWarp);
  }

private:
  unsigned numWarps;
  unsigned threadsPerWarp;
  ReassignEncodingOpHandlerList handlers; // TODO rename
  CoalescedConstraint constraint;
};

struct CoalescePass : public TritonGPUCoalesceBase<CoalescePass> {
  std::optional<CoalescedConstraint>
  getCoalescedConstraint(ModuleAxisInfoAnalysis &axisInfoAnalysis, Value ptr,
                         int numWarps, int threadsPerWarp) {
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

      // If there are no known contiguity properties, this value shouldn't
      // require any ordering, to avoid forcing any ordering.
      const auto &contiguity = queryAxisInfo(ptr).getContiguity();
      if (!std::any_of(contiguity.begin(), contiguity.end(),
                       [](int64_t x) { return x > 1; })) {
        return std::nullopt;
      }
      order = argSort(contiguity);
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

    auto constraint = std::min<int>(perThread, numElemsPerThread);
    return CoalescedConstraint{order, constraint};
  }

  Attribute getCoalescedEncoding(const CoalescedConstraint &constraint,
                                 Value ptr, int numWarps, int threadsPerWarp) {
    auto refType = ptr.getType();
    if (refType.isa<PointerType>())
      refType = refType.cast<PointerType>().getPointeeType();
    auto refTensorType = refType.cast<RankedTensorType>();

    SmallVector<unsigned, 4> sizePerThread(refTensorType.getRank(), 1);
    sizePerThread[constraint.order[0]] = constraint.sizePerThread;

    auto CTALayout = triton::gpu::getCTALayout(refTensorType.getEncoding());
    return triton::gpu::BlockedEncodingAttr::get(
        &getContext(), refTensorType.getShape(), sizePerThread,
        constraint.order, numWarps, threadsPerWarp, CTALayout);
  }

  std::function<Type(Type)>
  getTypeConverter(std::optional<CoalescedConstraint> encodingConstraint,
                   Value ptr, int numWarps, int threadsPerWarp) {
    if (encodingConstraint) {
      Attribute encoding = getCoalescedEncoding(*encodingConstraint, ptr,
                                                numWarps, threadsPerWarp);
      return [encoding](Type type) {
        RankedTensorType tensorType = type.cast<RankedTensorType>();
        return RankedTensorType::get(tensorType.getShape(),
                                     tensorType.getElementType(), encoding);
      };
    } else {
      return [](Type type) { return type; };
    }
  }

  bool typeObeysConstraint(Value ptr, const CoalescedConstraint &constraint) {
    auto tensorType = ptr.getType().dyn_cast<RankedTensorType>();
    if (!tensorType) {
      return false;
    }
    auto encoding =
        tensorType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
    if (!encoding) {
      return false;
    }
    if (!(encoding.getOrder() == ArrayRef<unsigned>(constraint.order))) {
      return false;
    }
    return encoding.getSizePerThread()[constraint.order[0]] >=
           constraint.sizePerThread;
  }

  template <class T>
  void coalesceOp(LayoutMap &layoutMap, Operation *op, Value ptr,
                  OpBuilder builder, int numWarps, int threadsPerWarp) {
    if (!layoutMap.count(ptr))
      return;

    // Convert operands
    // For load/store with tensor pointers, we don't have to change the
    // operands' type, we do this by changing the outputs' type of
    // `make_tensor_ptr`

    std::optional<CoalescedConstraint> encodingConstraint =
        layoutMap.lookup(ptr);

    // If there's no constraint for this pointer, do nothing
    if (!encodingConstraint) {
      return;
    }

    // If this type already obeys the constraints, do nothing.
    if (typeObeysConstraint(ptr, *encodingConstraint)) {
      return;
    }

    auto convertType =
        getTypeConverter(encodingConstraint, ptr, numWarps, threadsPerWarp);
    SmallVector<Value, 4> newArgs;

    // Convert output types
    SmallVector<Type, 4> newTypes;
    for (auto t : op->getResultTypes()) {
      bool isAsync = std::is_same<T, triton::gpu::InsertSliceAsyncOp>::value;
      newTypes.push_back(isAsync ? t : convertType(t));
    }

    for (auto operand : op->getOperands()) {
      auto tensorType = operand.getType().dyn_cast<RankedTensorType>();
      if (tensorType &&
          !tensorType.getEncoding().isa<triton::gpu::SharedEncodingAttr>()) {
        newArgs.push_back(builder.create<triton::gpu::ConvertLayoutOp>(
            op->getLoc(), convertType(tensorType), operand));
      } else
        newArgs.push_back(operand);
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
                                     Value ptr, OpBuilder builder, int numWarps,
                                     int threadsPerWarp) {
    if (!layoutMap.count(ptr))
      return;

    // Convert result type
    auto convertType =
        getTypeConverter(layoutMap.lookup(ptr), ptr, numWarps, threadsPerWarp);
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

  // TODO docstring. Something about sizePerThread and coalescing and limiting
  // layout conversions
  void tryResetDefaultEncoding(ModuleOp &moduleOp, const LayoutMap &layoutMap) {
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(moduleOp);
    int threadsPerWarp =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(moduleOp);

    // In the case that there's 1 unique order seen in the graph, we can be
    // fairly confident that resetting the default encoding should be
    // beneficial to reduce the number of layout conversions. However, if there
    // are multiple unique ops, a more sophisticated strategy might be needed
    // to assign layouts in a way to minimize layout conversions. For now, only
    // reset the default encoding if there's 1 unique order.
    std::optional<CoalescedConstraint> bestConstraint = std::nullopt;
    for (const auto &[_, constraint] : layoutMap) {
      if (!bestConstraint) {
        bestConstraint = constraint;
      } else if (constraint) {
        if (constraint->order != bestConstraint->order) {
          return;
        }
        bestConstraint->sizePerThread =
            std::max(constraint->sizePerThread, bestConstraint->sizePerThread);
      }
    }

    // If there's no constraints, there's no point to expand sizePerThread
    if (!bestConstraint) {
      return;
    }
    moduleOp.dump();

    ReassignEncodingOpHandlerList handlers;
    handlers.append<ReassignEncodingConstantOp>();
    // TODO: clean this up, e.g. maybe just pass lambdas...
    //       also some of the triton ops are duplicates.
    handlers.append<ReassignEncodingArithOps>();
    handlers.append<ReassignEncodingMathOps>();
    handlers.append<ReassignEncodingSameOpResultEncoding>();
    handlers.append<ReassignEncodingSupported<::mlir::gpu::BarrierOp>>();
    handlers.append<ReassignEncodingSupported<scf::ConditionOp>>();
    handlers.append<ReassignEncodingSupported<scf::ForOp>>();
    handlers.append<ReassignEncodingSupported<scf::IfOp>>();
    handlers.append<ReassignEncodingSupported<scf::WhileOp>>();
    handlers.append<ReassignEncodingSupported<scf::YieldOp>>();
    handlers.append<ReassignEncodingSupported<triton::AddPtrOp>>();
    handlers.append<ReassignEncodingSupported<triton::AssertOp>>();
    handlers.append<ReassignEncodingSupported<triton::AtomicRMWOp>>();
    handlers.append<ReassignEncodingSupported<triton::BitcastOp>>();
    handlers.append<ReassignEncodingSupported<triton::ExternElementwiseOp>>();
    handlers.append<ReassignEncodingSupported<triton::FuncOp>>();
    handlers.append<ReassignEncodingSupported<triton::GetProgramIdOp>>();
    handlers.append<ReassignEncodingSupported<triton::LoadOp>>();
    handlers.append<ReassignEncodingSupported<triton::MakeRangeOp>>();
    handlers.append<ReassignEncodingSupported<triton::ReturnOp>>();
    handlers.append<ReassignEncodingSupported<triton::SplatOp>>();
    handlers.append<ReassignEncodingSupported<triton::StoreOp>>();
    handlers.append<ReassignEncodingSupported<triton::gpu::CmpFOp>>();
    handlers.append<ReassignEncodingSupported<triton::gpu::CmpIOp>>();
    handlers.append<ReassignEncodingSupported<triton::gpu::SelectOp>>();
    handlers.append<ReassignEncodingSupported<ModuleOp>>();

    handlers.append<ReassignEncodingUnsupported<triton::BroadcastOp>>();
    handlers.append<ReassignEncodingUnsupported<triton::ExpandDimsOp>>();
    handlers.append<ReassignEncodingUnsupported<triton::ReduceOp>>();
    handlers.append<ReassignEncodingUnsupported<triton::ReduceReturnOp>>();
    handlers
        .append<ReassignEncodingUnsupported<triton::gpu::ConvertLayoutOp>>();

    ReassignEncodingSupportedState state;

    moduleOp.walk([&](Operation *curr) {
      if (!handlers.supported(curr, state)) {
        state.markUnsupported();
      }
    });

    if (!state.isSupported()) {
      return;
    }

    if (state.uniqueShapes() > 1) {
      return;
    }

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReassignEncodingPattern>(context, std::move(handlers),
                                          *bestConstraint, numWarps,
                                          threadsPerWarp);

    if (applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)).failed()) {
      std::cerr << " applyPatternsAndFoldGreedily failed" << std::endl;
      return signalPassFailure();
    } else {
      std::cerr << " Success! " << std::endl;
    }
    moduleOp.dump();
  }

  void runOnOperation() override {
    // Run axis info analysis
    ModuleOp moduleOp = getOperation();
    std::cerr
        << " runOnOperation: Coalesce, want to apply ReassignEncodingPattern"
        << std::endl;

    ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(moduleOp);
    int threadsPerWarp =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(moduleOp);

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
      layoutMap[ptr] = getCoalescedConstraint(axisInfoAnalysis, ptr, numWarps,
                                              threadsPerWarp);
    });

    tryResetDefaultEncoding(moduleOp, layoutMap);

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
        coalesceOp<triton::LoadOp>(layoutMap, curr, load.getPtr(), builder,
                                   numWarps, threadsPerWarp);
        return;
      }
      if (auto op = dyn_cast<triton::AtomicRMWOp>(curr)) {
        coalesceOp<triton::AtomicRMWOp>(layoutMap, curr, op.getPtr(), builder,
                                        numWarps, threadsPerWarp);
        return;
      }
      if (auto op = dyn_cast<triton::AtomicCASOp>(curr)) {
        coalesceOp<triton::AtomicCASOp>(layoutMap, curr, op.getPtr(), builder,
                                        numWarps, threadsPerWarp);
        return;
      }
      if (auto load = dyn_cast<triton::gpu::InsertSliceAsyncOp>(curr)) {
        coalesceOp<triton::gpu::InsertSliceAsyncOp>(
            layoutMap, curr, load.getSrc(), builder, numWarps, threadsPerWarp);
        return;
      }
      if (auto store = dyn_cast<triton::StoreOp>(curr)) {
        coalesceOp<triton::StoreOp>(layoutMap, curr, store.getPtr(), builder,
                                    numWarps, threadsPerWarp);
        return;
      }
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUCoalescePass() {
  return std::make_unique<CoalescePass>();
}
