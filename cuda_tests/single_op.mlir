func @reduce(%arg0: tensor<64x64xf32>) -> tensor<64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.reduce"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
        %2 = mhlo.add %arg1, %arg2: tensor<f32>
        "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<64x64xf32>, tensor<f32>) -> tensor<64xf32>
    return %1 : tensor<64xf32>
}

func @add(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
 %0 = mhlo.add %lhs, %rhs : tensor<4xf32>
 return %0 : tensor<4xf32>
}

func @sigmoid(%lhs: tensor<4xf32>) -> tensor<4xf32> {
 %0 = "mhlo.logistic"(%lhs) : (tensor<4xf32>) -> tensor<4xf32>
 return %0 : tensor<4xf32>
}

func @matmul(%lhs: tensor<64x64xf32>, %rhs: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %0: tensor<64x64xf32>
}

func @reshape(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x2xf32>) -> tensor<4xf32>
    %1 = "mhlo.reshape"(%arg1) : (tensor<2x2xf32>) -> tensor<4xf32>
    %2 = mhlo.add %0, %1 : tensor<4xf32>
    %3 = "mhlo.reshape"(%2) : (tensor<4xf32>) -> tensor<2x2xf32>
    return %3 : tensor<2x2xf32>
}