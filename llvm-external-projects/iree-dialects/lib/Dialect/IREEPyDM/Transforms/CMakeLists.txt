add_subdirectory(Optimize)
add_subdirectory(RTL)
add_subdirectory(ToIREE)

add_mlir_library(IREEDialectsIREEPyDMPasses
  Passes.cpp

  DEPENDS
  MLIRIREEPyDMTransformsPassesIncGen

  LINK_LIBS PUBLIC
  IREEDialectsIREEPyDMOptimizePasses
  IREEDialectsIREEPyDMRTLPasses
  IREEDialectsIREEPyDMToIREEPasses
  MLIRTransforms
)

iree_dialects_target_includes(IREEDialectsIREEPyDMPasses)
