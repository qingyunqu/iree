// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/Utils.h"

#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace iree_compiler {

bool isEntryPoint(FuncOp func) { return func.isPublic(); }

IREE::HAL::ExecutableEntryPointOp getEntryPoint(FuncOp funcOp) {
  auto variantOp = funcOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  for (auto op : variantOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
    if (op.sym_name() == funcOp.getName()) {
      return op;
    }
  }
  return nullptr;
}

llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> getAllEntryPoints(
    ModuleOp module) {
  auto variantOp = module->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPointOps;
  for (auto op : variantOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
    entryPointOps[op.sym_name()] = op;
  }
  return entryPointOps;
}

IREE::HAL::TranslationInfo getTranslationInfo(FuncOp funcOp) {
  auto entryPointOp = getEntryPoint(funcOp);
  if (!entryPointOp) return nullptr;
  return getTranslationInfo(entryPointOp);
}

void setTranslationInfo(FuncOp entryPointFn,
                        IREE::HAL::DispatchLoweringPassPipeline passPipeline,
                        ArrayRef<int64_t> workgroupSize,
                        ArrayRef<int64_t> workloadPerWorkgroup) {
  auto entryPointOp = getEntryPoint(entryPointFn);
  auto translationInfo = buildTranslationInfo(
      passPipeline, workloadPerWorkgroup, entryPointFn.getContext());
  setTranslationInfo(entryPointOp, translationInfo, workgroupSize);
}

SmallVector<unsigned> getPartitionedLoops(Operation *op) {
  if (auto mmt4dOp = dyn_cast<linalg::Mmt4DOp>(op)) {
    return {0, 1};
  }
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    SmallVector<unsigned> partitionedLoops;
    for (auto indexedIterator : llvm::enumerate(linalgOp.iterator_types())) {
      if (isParallelIterator(indexedIterator.value())) {
        partitionedLoops.push_back(indexedIterator.index());
      }
    }
    // Only keep the last kNumMaxParallelDims if we have more than that.
    while (partitionedLoops.size() > kNumMaxParallelDims) {
      partitionedLoops.erase(partitionedLoops.begin());
    }
    return partitionedLoops;
  }
  if (auto tilableOp = dyn_cast<linalg_ext::TiledOpInterface>(op)) {
    return tilableOp.getPartitionableLoops(kNumMaxParallelDims);
  }
  return {};
}

LogicalResult setOpConfigAndEntryPointFnTranslation(
    FuncOp entryPointFn, Operation *op, IREE::HAL::LoweringConfig config,
    IREE::HAL::DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workgroupSize) {
  auto partitionedLoops = getPartitionedLoops(op);
  SmallVector<int64_t, 3> workloadPerWorkgroup;
  auto tileSizes = getTileSizes(config, 0);
  if (!tileSizes.empty() && !partitionedLoops.empty()) {
    for (unsigned depth : partitionedLoops) {
      if (depth >= tileSizes.size()) {
        return op->emitOpError(
                   "illegal configuration for lowering op, expect first level "
                   "tile size to contain at least ")
               << partitionedLoops.back() << " elements";
      }
      if (tileSizes[depth] == 0) {
        return op->emitOpError("illegal to set tilesize of loop ")
               << depth
               << " to zero since it is set to be partitioned at the flow "
                  "level";
      }
      workloadPerWorkgroup.push_back(tileSizes[depth]);
    }
    if (!workloadPerWorkgroup.empty()) {
      workloadPerWorkgroup =
          llvm::to_vector<3>(llvm::reverse(workloadPerWorkgroup));
    }
  }
  auto entryPointOp = getEntryPoint(entryPointFn);
  if (!entryPointOp) {
    return entryPointFn.emitOpError(
        "unable to find entry point op for entry point function");
  }
  IREE::HAL::TranslationInfo translationInfo = buildTranslationInfo(
      passPipeline, workloadPerWorkgroup, entryPointOp->getContext());
  setTranslationInfo(entryPointOp, translationInfo, workgroupSize);
  return success();
}

/// Walk up the defs of the view, to get the untiled value. Either walks up
/// `ViewOpInterface` op-chains or the `subtensor` op-chains.
static Value getViewSource(Value view) {
  while (true) {
    Operation *definingOp = view.getDefiningOp();
    if (!definingOp) break;
    if (auto viewOp = view.getDefiningOp<ViewLikeOpInterface>()) {
      view = viewOp.getViewSource();
      continue;
    }
    if (auto subTensorOp = view.getDefiningOp<tensor::ExtractSliceOp>()) {
      view = subTensorOp.source();
      continue;
    }
    if (auto dispatchTensorLoadOp =
            view.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>()) {
      view = dispatchTensorLoadOp.source();
      continue;
    }
    break;
  }
  return view;
}

Type getUntiledType(Value tiledView) {
  Value viewSource = getViewSource(tiledView);
  return viewSource.getType();
}

ArrayRef<int64_t> getUntiledShape(Value tiledView) {
  auto type = getUntiledType(tiledView);
  return TypeSwitch<Type, ArrayRef<int64_t>>(type)
      .Case<ShapedType, IREE::Flow::DispatchTensorType>(
          [&](auto shapedType) { return shapedType.getShape(); })
      .Default([&](Type type) { return ArrayRef<int64_t>{}; });
}

/// Returns the untiled shape of the output of a `LinalgOp`.
// TODO(ravishankarm): Using the result shape for vectorization should be
// avoided. Ideally the tile size is enough. But there is a phase ordering issue
// which prevents the tile size from being known at this point.
ArrayRef<int64_t> getUntiledResultShape(linalg::LinalgOp linalgOp,
                                        unsigned resultNum) {
  // Check the shape of the `outs` operand.
  auto outputShape = getUntiledShape(linalgOp.outputs()[resultNum]);
  if (llvm::none_of(outputShape, ShapedType::isDynamic)) return outputShape;

  // For Linalg ops with buffer semantics, there won't exist op results and
  // hence IR users. Also directly return.
  if (linalgOp.hasBufferSemantics()) return outputShape;

  // Try to use the result value and check if the untiled shape can be obtained
  // based on the uses.
  Value result = linalgOp->getResult(resultNum);
  for (Operation *user : result.getUsers()) {
    if (auto storeOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(user)) {
      return storeOp.target()
          .getType()
          .cast<IREE::Flow::DispatchTensorType>()
          .getShape();
    }
  }
  return result.getType().cast<ShapedType>().getShape();
}

//===----------------------------------------------------------------------===//
// Get the tiled loops in the computation.
//===----------------------------------------------------------------------===//

/// Returns the first of `exprs` which is of the type `T`.
template <typename T>
static AffineExpr getAffineExprOfType(ArrayRef<AffineExpr> exprs) {
  for (auto expr : exprs) {
    if (expr.isa<T>()) return expr;
  }
  return nullptr;
}

/// Returns true if the `expr` is on of the types in {`T1`, `T2`, `T3...`}.
template <typename T>
static bool isaAffineExprOfType(AffineExpr expr) {
  return expr.isa<T>();
}
template <typename T1, typename T2, typename... T3>
static bool isaAffineExprOfType(AffineExpr expr) {
  if (expr.isa<T1>()) {
    return true;
  }
  return isaAffineExprOfType<T2, T3...>(expr);
}

/// Returns a Value that repreesnts the value for symbol or dim expr for the map
/// in the `applyOp`.
static Value getValueForDimOrSymbol(AffineApplyOp applyOp, AffineExpr expr) {
  unsigned numDims = applyOp.getAffineMap().getNumDims();
  if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
    return applyOp.getOperand(dimExpr.getPosition());
  }
  if (auto symbolExpr = expr.dyn_cast<AffineSymbolExpr>()) {
    return applyOp.getOperand(numDims + symbolExpr.getPosition());
  }
  return nullptr;
}
static SmallVector<Value> getValuesForDimsOrSymbols(
    AffineApplyOp applyOp, ArrayRef<AffineExpr> exprs) {
  SmallVector<Value> vals;
  for (auto expr : exprs) {
    vals.push_back(getValueForDimOrSymbol(applyOp, expr));
  }
  return vals;
}

/// Returns the dimension for any operation that has the `dimension`
/// attribute. Currently tested for `flow.dispatch.workgroup.(size|count|id)`,
/// `hal.interface.workgroup.(size|count|id)`.
template <typename T>
static Optional<unsigned> getDimension(Operation *op) {
  if (auto tOp = dyn_cast<T>(op)) {
    return tOp.dimension().getZExtValue();
  }
  return llvm::None;
}
template <typename T1, typename T2, typename... T3>
static Optional<unsigned> getDimension(Operation *op) {
  if (!op) return llvm::None;
  if (auto dimension = getDimension<T1>(op)) {
    return dimension;
  }
  return getDimension<T2, T3...>(op);
}

/// Checks that all `vals` are defined by either
/// `flow.dispatch.workgroup.(size|count|id)` or
/// `hal.interface.workgroup.(size|count|id)` using the same `dimension`. If any
/// element of `vals` is not defined by one of these ops, or the dimensions dont
/// match, returns llvm::None. On success returns the dimension.  If
/// `refDimension` is passed checks if the dimension matches the given value.
template <typename... T>
static Optional<unsigned> checkDimensions(
    ArrayRef<Value> vals, Optional<unsigned> refDimension = llvm::None) {
  for (auto v : vals) {
    auto currDimension = getDimension<T...>(v.getDefiningOp());
    if (!currDimension) return llvm::None;
    if (refDimension) {
      if (refDimension.getValue() != currDimension.getValue()) {
        return llvm::None;
      }
    } else {
      refDimension = currDimension.getValue();
    }
  }
  return refDimension;
}

namespace {
/// Visitor to walk `lb` of a distributed loop. Expected the expression to be of
/// the form `a + b * c`, where `a` is the original `lb` and `b`, `c` are either
/// hal.interface.workgroup.id or hal.interface.workgroup.size.
class LowerBoundExprVisitor
    : public AffineExprVisitor<LowerBoundExprVisitor, LogicalResult> {
 public:
  LowerBoundExprVisitor(AffineApplyOp applyOp, TiledLoopInfo &loopInfo)
      : applyOp(applyOp), loopInfo(loopInfo) {}

  LogicalResult visitSymbolExpr(AffineSymbolExpr /*expr*/) { return failure(); }
  LogicalResult visitDimExpr(AffineDimExpr /*expr*/) { return failure(); }
  LogicalResult visitConstantExpr(AffineConstantExpr /*expr*/) {
    return failure();
  }
  LogicalResult visitAffineBinaryOpExpr(AffineBinaryOpExpr /*expr*/) {
    return failure();
  }

  LogicalResult visitAddExpr(AffineBinaryOpExpr expr) {
    AffineExpr offsetExpr =
        getAffineExprOfType<AffineBinaryOpExpr>({expr.getLHS(), expr.getRHS()});
    if (!offsetExpr) {
      // One of the expressions has to be a binary op expr.
      return failure();
    }
    // The other expression must be the undistributed `lb`.
    AffineExpr lbExpr =
        (offsetExpr == expr.getLHS() ? expr.getRHS() : expr.getLHS());
    if (isaAffineExprOfType<AffineDimExpr, AffineSymbolExpr>(lbExpr)) {
      Value v = getValueForDimOrSymbol(applyOp, lbExpr);
      if (!v) {
        return failure();
      }
      loopInfo.lb = getAsOpFoldResult(v);
    } else if (auto constExpr = lbExpr.dyn_cast<AffineConstantExpr>()) {
      loopInfo.lb = IntegerAttr::get(IndexType::get(applyOp.getContext()),
                                     constExpr.getValue());
    } else {
      return failure();
    }
    return visit(offsetExpr);
  }

  LogicalResult visitMulExpr(AffineBinaryOpExpr expr) {
    SmallVector<Value> vals =
        getValuesForDimsOrSymbols(applyOp, {expr.getLHS(), expr.getRHS()});
    if (vals.size() != 2 || !vals[0] || !vals[1]) {
      return failure();
    }
    Optional<unsigned> dimension =
        checkDimensions<IREE::HAL::InterfaceWorkgroupIDOp,
                        IREE::HAL::InterfaceWorkgroupSizeOp>(vals);
    if (!dimension) {
      return failure();
    }
    loopInfo.distributionDim = dimension.getValue();
    if (!loopInfo.lb) {
      loopInfo.lb = IntegerAttr::get(IndexType::get(applyOp.getContext()), 0);
    }
    return success();
  }

 private:
  AffineApplyOp applyOp;
  TiledLoopInfo &loopInfo;
};

/// Visitor to walk the `step` of a distributed loop. Expected the expression to
/// be of the form `a * b * c`, where they could be the dynamic `step` or
/// defined by `hal.interface.workgroup.size`/`hal.interface.workgroup.count`
/// operation.
class StepExprVisitor
    : public AffineExprVisitor<StepExprVisitor, LogicalResult> {
 public:
  StepExprVisitor(AffineApplyOp applyOp, TiledLoopInfo &loopInfo)
      : applyOp(applyOp), loopInfo(loopInfo) {}

  LogicalResult visitSymbolExpr(AffineSymbolExpr /*expr*/) { return failure(); }
  LogicalResult visitDimExpr(AffineDimExpr /*expr*/) { return failure(); }
  LogicalResult visitConstantExpr(AffineConstantExpr /*expr*/) {
    return failure();
  }
  LogicalResult visitAffineBinaryOpExpr(AffineBinaryOpExpr /*expr*/) {
    return failure();
  }

  LogicalResult visitMulExpr(AffineBinaryOpExpr expr) {
    // Check if one of the operands is a binary op expr.
    SmallVector<AffineExpr> sentinels;
    if (auto e = getAffineExprOfType<AffineBinaryOpExpr>(
            {expr.getLHS(), expr.getRHS()})) {
      AffineExpr otherExpr =
          (e == expr.getLHS() ? expr.getRHS() : expr.getLHS());
      if (failed(processSentinel(otherExpr, sentinels))) {
        return failure();
      }
      expr = e.cast<AffineBinaryOpExpr>();
    } else {
      loopInfo.step = IntegerAttr::get(IndexType::get(applyOp.getContext()), 1);
    }

    if (failed(processSentinel(expr.getLHS(), sentinels)) ||
        failed(processSentinel(expr.getRHS(), sentinels))) {
      return failure();
    }
    // Either there are 3 sentinels and step isnt set, or there are two
    // sentinels and the step is set.
    if (sentinels.size() == 3) {
      if (loopInfo.step) {
        return failure();
      }
      auto it = sentinels.begin();
      for (auto ie = sentinels.end(); it != ie; ++it) {
        Value v = getValueForDimOrSymbol(applyOp, *it);
        if (!v.getDefiningOp<IREE::HAL::InterfaceWorkgroupSizeOp>() &&
            !v.getDefiningOp<IREE::HAL::InterfaceWorkgroupCountOp>()) {
          loopInfo.step = getAsOpFoldResult(v);
          break;
        }
      }
      if (it != sentinels.end()) {
        sentinels.erase(it);
      }
    }

    if (sentinels.size() != 2 || !loopInfo.step) {
      return failure();
    }
    if (!checkDimensions<IREE::HAL::InterfaceWorkgroupCountOp,
                         IREE::HAL::InterfaceWorkgroupSizeOp>(
            getValuesForDimsOrSymbols(applyOp, sentinels),
            loopInfo.distributionDim)) {
      return failure();
    }
    return success();
  }

 private:
  LogicalResult processSentinel(AffineExpr e,
                                SmallVectorImpl<AffineExpr> &sentinels) {
    if (isaAffineExprOfType<AffineDimExpr, AffineSymbolExpr>(e)) {
      sentinels.push_back(e);
      return success();
    } else if (auto constExpr = e.dyn_cast<AffineConstantExpr>()) {
      if (loopInfo.step) {
        return failure();
      }
      loopInfo.step = IntegerAttr::get(IndexType::get(applyOp.getContext()),
                                       constExpr.getValue());
      return success();
    }
    return failure();
  }

  AffineApplyOp applyOp;
  TiledLoopInfo &loopInfo;
};
}  // namespace

/// Checks if the `forOp` is a tiled + distributed op. Looks for the op of this
/// form
/// ```
///   %dim = arith.constant ... : index
///   %id = flow.dispatch.workgroup.id[%dim]
///   %count = flow.dispatch.workgroup.count[%dim]
///   %size = flow.dispatch.workgroup.size[%dim]
///   %offset = affine.apply affine_map<(d0)[s0, s1] -> (d0 + s0 *
///   s1)>(%lb)[%id, %size] %new_step = affine.apply affine_map<(d0)[s0, s1] ->
///   (d0 * s0 * s1)>(%step)[%id, %size] scf.for %iv = %offset to %ub step
///   %new_step {
///     ...
///   }
/// ```
static Optional<TiledLoopInfo> isTiledLoop(MLIRContext *context,
                                           scf::ForOp forOp) {
  TiledLoopInfo loopInfo;
  auto lbApplyOp = forOp.lowerBound().getDefiningOp<AffineApplyOp>();
  if (!lbApplyOp) {
    return llvm::None;
  }
  LowerBoundExprVisitor lbVisitor(lbApplyOp, loopInfo);
  auto stepApplyOp = forOp.step().getDefiningOp<AffineApplyOp>();
  if (!stepApplyOp) {
    return llvm::None;
  }
  StepExprVisitor stepVisitor(stepApplyOp, loopInfo);
  if (failed(lbVisitor.visit(lbApplyOp.getAffineMap().getResults()[0])) ||
      failed(stepVisitor.visit(stepApplyOp.getAffineMap().getResults()[0]))) {
    return llvm::None;
  }
  if (!loopInfo.lb || !loopInfo.step) {
    return llvm::None;
  }
  loopInfo.ub = getAsOpFoldResult(forOp.upperBound());
  return loopInfo;
}

LogicalResult getFilteredOps(FuncOp funcOp, RootOpFilteringFn filteringFn,
                             SmallVectorImpl<Operation *> &filteredOps,
                             SmallVectorImpl<TiledLoopInfo> &tiledLoops) {
  Region &region = funcOp.body();
  if (!llvm::hasSingleElement(region)) {
    return funcOp.emitError("unable dispatch function with multiple blocks");
  }
  Block *body = &region.front();
  MLIRContext *context = funcOp.getContext();
  auto forOps = body->getOps<scf::ForOp>();
  while (!forOps.empty()) {
    if (!llvm::hasSingleElement(forOps)) return failure();
    scf::ForOp forOp = *(forOps.begin());
    if (auto tiledLoopInfo = isTiledLoop(context, forOp)) {
      tiledLoops.emplace_back(std::move(tiledLoopInfo.getValue()));
    }
    body = forOp.getBody();
    forOps = body->getOps<scf::ForOp>();
  }
  for (Operation &op : body->getOperations()) {
    if (filteringFn(&op)) {
      filteredOps.push_back(&op);
    }
  }
  return success();
}

LogicalResult getComputeOps(FuncOp funcOp,
                            SmallVectorImpl<Operation *> &computeOps,
                            SmallVectorImpl<TiledLoopInfo> &tiledLoops) {
  if (failed(getFilteredOps(
          funcOp,
          [](Operation *op) {
            return isa<linalg::LinalgOp, linalg_ext::TiledOpInterface>(op);
          },
          computeOps, tiledLoops))) {
    return failure();
  }

  // Propagate markers to all ops. If one of the ops has a marker all ops in
  // this loop need to have marker since body of the loop maps to a workgroup.
  // TODO(ravishankarm): Temporary WAR till a better story w.r.t markers is
  // figured out.
  Optional<StringRef> marker = llvm::None;
  for (auto op : computeOps) {
    if (hasMarker(op)) {
      assert((!marker || marker.getValue() == getMarkerOrNull(op)) &&
             "expected all markers within op to be the same");
      marker = getMarkerOrNull(op);
    }
  }
  if (!marker.hasValue() && !tiledLoops.empty()) {
    marker = getWorkgroupMarker();
  }
  if (marker.hasValue()) {
    for (auto op : computeOps) {
      setMarker(op, marker.getValue());
    }
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
