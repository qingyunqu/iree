// RUN: iree-opt -split-input-file -iree-flow-outline-large-constants %s | IreeFileCheck %s

// CHECK: util.global private @[[LARGE_VARIABLE:.+]] {noinline} = dense<1.200000e+00> : tensor<512x128xf32>
func @fn1() -> (tensor<2xf32>, tensor<512x128xf32>) {
  // CHECK-DAG: %[[SMALL_VALUE:.+]] = arith.constant dense<{{.+}}> : tensor<2xf32>
  %cst_0 = arith.constant dense<[0.0287729427, 0.0297581609]> : tensor<2xf32>
  // CHECK-DAG: %[[LARGE_VALUE:.+]] = util.global.load @[[LARGE_VARIABLE]] : tensor<512x128xf32>
  %cst_1 = arith.constant dense<1.2> : tensor<512x128xf32>
  return %cst_0, %cst_1 : tensor<2xf32>, tensor<512x128xf32>
}
