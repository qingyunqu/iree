func @add_reduce(%arg0: tensor<64x64xf32>, %arg3: tensor<64x64xf32>) -> tensor<64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = mhlo.add %arg0, %arg3 : tensor<64x64xf32>
    %1 = "mhlo.reduce"(%3, %0) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
        %2 = mhlo.add %arg1, %arg2: tensor<f32>
        "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<64x64xf32>, tensor<f32>) -> tensor<64xf32>
    return %1 : tensor<64xf32>
}

func @reduce_add(%arg0: tensor<64x64xf32>, %arg3: tensor<64xf32>) -> tensor<64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.reduce"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
        %2 = mhlo.add %arg1, %arg2: tensor<f32>
        "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<64x64xf32>, tensor<f32>) -> tensor<64xf32>
    %2 = mhlo.add %1, %arg3 : tensor<64xf32>
    return %2 : tensor<64xf32>
}

func @wide_add_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>) -> tensor<4xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4xf32>
    %1 = mhlo.add %arg2, %arg3 : tensor<4xf32>
    %2 = mhlo.multiply %0, %1 : tensor<4xf32>
    return %2 : tensor<4xf32>
}

// func @add_and_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>) -> tuple<tensor<4xf32>, tensor<4xf32>> {
//     %0 = mhlo.add %arg0, %arg1 : tensor<4xf32>
//     %1 = mhlo.multiply %arg2, %arg3 : tensor<4xf32>
//     %2 = "mhlo.tuple"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tuple<tensor<4xf32>, tensor<4xf32>>
//     return %2 : tuple<tensor<4xf32>, tensor<4xf32>>
// }

func @add_and_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    %0 = mhlo.add %arg0, %arg1 : tensor<4xf32>
    %1 = mhlo.multiply %arg2, %arg3 : tensor<4xf32>
    return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

func @wide_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>, %arg3: tensor<4xf32>) -> tensor<8xf32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<4xf32>
    %1 = mhlo.add %arg2, %arg3 : tensor<4xf32>
    %2 = "mhlo.concatenate"(%0, %1) {dimension = 0 : i64} : (tensor<4xf32>, tensor<4xf32>) -> tensor<8xf32>
    return %2 : tensor<8xf32>
}

func @sigmoid_add(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
 %0 = "mhlo.logistic"(%lhs) : (tensor<4xf32>) -> tensor<4xf32>
 %1 = "mhlo.add"(%0, %rhs) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
 return %1 : tensor<4xf32>
}

func @matmul_add(%lhs: tensor<64x64xf32>, %rhs: tensor<64x64xf32>, %bias: tensor<64xf32>) -> tensor<64x64xf32> {
    %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %1 = "mhlo.broadcast_in_dim"(%bias) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<64xf32>) -> (tensor<64x64xf32>)
    %2 = mhlo.add %0, %1 : tensor<64x64xf32>
    return %2: tensor<64x64xf32>
}

func @fusion(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "mhlo.fusion"(%lhs, %rhs) ( {
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):  // no predecessors
      %1 = "mhlo.logistic"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
      %2= "mhlo.add"(%1, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "mhlo.return"(%2) : (tensor<4xf32>) -> ()
    }) {fusion_kind = "kLoop"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}