// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

//====---------------------------------------------------------------------===//
// Patterns for vectorization
//====---------------------------------------------------------------------===//

static void populateVectorizationPatterns(RewritePatternSet &patterns) {
  linalg::insertVectorizationPatterns<linalg::FillOp, linalg::CopyOp,
                                      linalg::GenericOp,
                                      linalg::ContractionOpInterface>(
      patterns, linalg::LinalgVectorizationOptions(),
      linalg::LinalgTransformationFilter(
          Identifier::get(getVectorizeMarker(), patterns.getContext())));
}

static Optional<SmallVector<int64_t, 4>> getGPUNativeVectorSize(Operation *op) {
  if ((OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)) {
    if (auto vecType = op->getResultTypes()[0].dyn_cast<VectorType>()) {
      // Map elementwise ops to vec4.
      SmallVector<int64_t, 4> nativeSize(vecType.getRank(), 1);
      nativeSize.back() = 4;
      return nativeSize;
    }
  } else if (auto vt = dyn_cast<VectorTransferOpInterface>(op)) {
    auto rank = vt.getVectorType().getRank();
    SmallVector<int64_t, 4> nativeSize(rank, 1);
    // Load 4 elements on the most inner dimension.
    for (auto dim : llvm::enumerate(vt.permutation_map().getResults())) {
      if (auto dimExpr = dim.value().dyn_cast<AffineDimExpr>()) {
        if (dimExpr.getPosition() == vt.permutation_map().getNumDims() - 1)
          nativeSize[dim.index()] = 4;
      }
    }
    return nativeSize;
  } else if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
    unsigned lastParalleldim = 0;
    for (auto it : llvm::enumerate(contract.iterator_types())) {
      if (isParallelIterator(it.value())) lastParalleldim = it.index();
    }
    SmallVector<int64_t, 4> nativeSize(contract.iterator_types().size(), 1);
    nativeSize[lastParalleldim] = 4;
    return nativeSize;
  }
  return llvm::None;
}

static void populateVectorUnrollPatterns(RewritePatternSet &patterns) {
  vector::populateVectorUnrollPatterns(
      patterns,
      vector::UnrollVectorOptions().setNativeShapeFn(getGPUNativeVectorSize));
}

namespace {
struct LLVMGPUVectorizationPass
    : public LLVMGPUVectorizationBase<LLVMGPUVectorizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    {
      // Step 1. Vectorize
      RewritePatternSet vectorizationPatterns(context);
      populateVectorizationPatterns(vectorizationPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(vectorizationPatterns));

      // Fold consumer add ops into the contraction op itself.
      RewritePatternSet canonicalizationPatterns(context);
      vector::ContractionOp::getCanonicalizationPatterns(
          canonicalizationPatterns, context);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(canonicalizationPatterns));

      RewritePatternSet vectorUnrollPatterns(context);
      populateVectorUnrollPatterns(vectorUnrollPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(vectorUnrollPatterns));
    }
    {
      // Step 2. Lower transfer op to canonical form.
      RewritePatternSet lowerTransferOpPatterns(funcOp.getContext());
      vector::populateVectorToVectorCanonicalizationPatterns(
          lowerTransferOpPatterns);
      vector::populateVectorTransferPermutationMapLoweringPatterns(
          lowerTransferOpPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(lowerTransferOpPatterns));
    }

    {
      // Step 2. Unroll the vetors to native size and canonicalize.
      RewritePatternSet vectorUnrollPatterns(context);
      populateVectorUnrollPatterns(vectorUnrollPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(vectorUnrollPatterns));

      RewritePatternSet canonicalizationPatterns(funcOp.getContext());
      vector::populateVectorToVectorCanonicalizationPatterns(
          canonicalizationPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(canonicalizationPatterns));
    }
    {
      // Step 3. Lower contract op to outer product.
      RewritePatternSet contractLoweringPatterns(funcOp.getContext());
      vector::populateVectorBroadcastLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorContractLoweringPatterns(
          contractLoweringPatterns,
          vector::VectorTransformsOptions().setVectorTransformsOptions(
              vector::VectorContractLowering::OuterProduct));
      vector::populateVectorMaskOpLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorShapeCastLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorMultiReductionLoweringPatterns(
          contractLoweringPatterns);
      (void)applyPatternsAndFoldGreedily(funcOp,
                                         std::move(contractLoweringPatterns));
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLLVMGPUVectorizationPass() {
  return std::make_unique<LLVMGPUVectorizationPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
