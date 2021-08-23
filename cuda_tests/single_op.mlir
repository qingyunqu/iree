func @reduce(%arg0: tensor<64x64xf32>) -> tensor<64xf32> {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = "mhlo.reduce"(%arg0, %0) ( {
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):  // no predecessors
        %2 = mhlo.add %arg1, %arg2: tensor<f32>
        "mhlo.return"(%2) : (tensor<f32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<64x64xf32>, tensor<f32>) -> tensor<64xf32>
    return %1 : tensor<64xf32>
}

// func @reduce2(%arg0: tensor<392x256xf32>) -> (tensor<392xf32>, tensor<392xf32>) {
//     %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
//     %4 = "mhlo.reduce"(%arg0, %2) ( {
//     ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
//       %25 = mhlo.add %arg3, %arg4 : tensor<f32>
//       "mhlo.return"(%25) : (tensor<f32>) -> ()
//     }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<392x256xf32>, tensor<f32>) -> tensor<392xf32>
//     %6 = mhlo.multiply %arg0, %arg0 : tensor<392x256xf32>
//     %7 = "mhlo.reduce"(%6, %2) ( {
//     ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
//       %25 = mhlo.add %arg3, %arg4 : tensor<f32>
//       "mhlo.return"(%25) : (tensor<f32>) -> ()
//     }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<392x256xf32>, tensor<f32>) -> tensor<392xf32>
//     return %4, %7 : tensor<392xf32>, tensor<392xf32>
// }

func @reduce2(%arg0: tensor<392x256xf32>) -> (tensor<392xf32>, tensor<392xf32>) {
    %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %6 = mhlo.multiply %arg0, %arg0 : tensor<392x256xf32>
    %7:2 = "mhlo.reduce"(%arg0, %6, %2, %3) ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>):  // no predecessors
      %25 = mhlo.add %arg3, %arg4 : tensor<f32>
      %26 = mhlo.add %arg2, %arg5 : tensor<f32>
      "mhlo.return"(%26, %25) : (tensor<f32>, tensor<f32>) -> ()
    }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<392x256xf32>, tensor<392x256xf32>, tensor<f32>, tensor<f32>) -> (tensor<392xf32>, tensor<392xf32>)
    return %7#0, %7#1 : tensor<392xf32>, tensor<392xf32>
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