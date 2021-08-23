builtin.module  {
  builtin.func @batch_norm_inference(%arg0: tensor<2x14x14x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<2x14x14x256xf32> {
    %0 = mhlo.constant dense<1.000000e-03> : tensor<256xf32>
    %1 = "mhlo.fusion"(%arg4) ( {
    ^bb0(%arg5: tensor<256xf32>):  // no predecessors
      %3 = mhlo.add %arg5, %0 : tensor<256xf32>
      %4 = "mhlo.rsqrt"(%3) : (tensor<256xf32>) -> tensor<256xf32>
      "mhlo.return"(%4) : (tensor<256xf32>) -> ()
    }) {fusion_kind = "kLoop"} : (tensor<256xf32>) -> tensor<256xf32>
    %2 = "mhlo.fusion"(%arg2, %arg3, %arg0, %arg1, %1) ( {
    ^bb0(%arg5: tensor<256xf32>, %arg6: tensor<256xf32>, %arg7: tensor<2x14x14x256xf32>, %arg8: tensor<256xf32>, %arg9: tensor<256xf32>):  // no predecessors
      %3 = mhlo.multiply %arg8, %arg9 : tensor<256xf32>
      %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<2x14x14x256xf32>
      %5 = mhlo.multiply %arg7, %4 : tensor<2x14x14x256xf32>
      %6 = mhlo.multiply %arg6, %3 : tensor<256xf32>
      %7 = mhlo.subtract %arg5, %6 : tensor<256xf32>
      %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<2x14x14x256xf32>
      %9 = mhlo.add %5, %8 : tensor<2x14x14x256xf32>
      "mhlo.return"(%9) : (tensor<2x14x14x256xf32>) -> ()
    }) {fusion_kind = "kLoop"} : (tensor<256xf32>, tensor<256xf32>, tensor<2x14x14x256xf32>, tensor<256xf32>, tensor<256xf32>) -> tensor<2x14x14x256xf32>
    return %2 : tensor<2x14x14x256xf32>
  }
  builtin.func @layer_norm_inference(%arg0: tensor<2x14x14x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<2x14x14x256xf32> {
    %0 = mhlo.constant dense<1.000000e-03> : tensor<392xf32>
    %1 = mhlo.constant dense<3.906250e-03> : tensor<392xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "mhlo.fusion"(%arg0) ( {
    ^bb0(%arg3: tensor<2x14x14x256xf32>):  // no predecessors
      %6 = "mhlo.reshape"(%arg3) : (tensor<2x14x14x256xf32>) -> tensor<1x392x256x1xf32>
      %7 = "mhlo.reduce"(%6, %2) ( {
      ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):  // no predecessors
        %8 = mhlo.add %arg4, %arg5 : tensor<f32>
        "mhlo.return"(%8) : (tensor<f32>) -> ()
      }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<1x392x256x1xf32>, tensor<f32>) -> tensor<392xf32>
      "mhlo.return"(%7) : (tensor<392xf32>) -> ()
    }) {fusion_kind = "kLoop"} : (tensor<2x14x14x256xf32>) -> tensor<392xf32>
    %4 = "mhlo.fusion"(%3, %arg0) ( {
    ^bb0(%arg3: tensor<392xf32>, %arg4: tensor<2x14x14x256xf32>):  // no predecessors
      %6 = "mhlo.reshape"(%arg4) : (tensor<2x14x14x256xf32>) -> tensor<1x392x256x1xf32>
      %7 = mhlo.multiply %6, %6 : tensor<1x392x256x1xf32>
      %8 = "mhlo.reduce"(%7, %2) ( {
      ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):  // no predecessors
        %15 = mhlo.add %arg5, %arg6 : tensor<f32>
        "mhlo.return"(%15) : (tensor<f32>) -> ()
      }) {dimensions = dense<[0, 2, 3]> : tensor<3xi64>} : (tensor<1x392x256x1xf32>, tensor<f32>) -> tensor<392xf32>
      %9 = mhlo.multiply %8, %1 : tensor<392xf32>
      %10 = mhlo.multiply %arg3, %1 : tensor<392xf32>
      %11 = mhlo.multiply %10, %10 : tensor<392xf32>
      %12 = mhlo.subtract %9, %11 : tensor<392xf32>
      %13 = mhlo.add %12, %0 : tensor<392xf32>
      %14 = "mhlo.rsqrt"(%13) : (tensor<392xf32>) -> tensor<392xf32>
      "mhlo.return"(%14) : (tensor<392xf32>) -> ()
    }) {fusion_kind = "kLoop"} : (tensor<392xf32>, tensor<2x14x14x256xf32>) -> tensor<392xf32>
    %5 = "mhlo.fusion"(%arg2, %arg1, %4, %3, %arg0) ( {
    ^bb0(%arg3: tensor<256xf32>, %arg4: tensor<256xf32>, %arg5: tensor<392xf32>, %arg6: tensor<392xf32>, %arg7: tensor<2x14x14x256xf32>):  // no predecessors
      %6 = "mhlo.reshape"(%arg7) : (tensor<2x14x14x256xf32>) -> tensor<1x392x256x1xf32>
      %7 = mhlo.multiply %arg6, %1 : tensor<392xf32>
      %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<392xf32>) -> tensor<1x392x256x1xf32>
      %9 = mhlo.subtract %6, %8 : tensor<1x392x256x1xf32>
      %10 = "mhlo.broadcast_in_dim"(%arg5) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<392xf32>) -> tensor<1x392x256x1xf32>
      %11 = mhlo.multiply %9, %10 : tensor<1x392x256x1xf32>
      %12 = "mhlo.reshape"(%11) : (tensor<1x392x256x1xf32>) -> tensor<2x14x14x256xf32>
      %13 = "mhlo.broadcast_in_dim"(%arg4) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<2x14x14x256xf32>
      %14 = mhlo.multiply %12, %13 : tensor<2x14x14x256xf32>
      %15 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<2x14x14x256xf32>
      %16 = mhlo.add %14, %15 : tensor<2x14x14x256xf32>
      "mhlo.return"(%16) : (tensor<2x14x14x256xf32>) -> ()
    }) {fusion_kind = "kLoop"} : (tensor<256xf32>, tensor<256xf32>, tensor<392xf32>, tensor<392xf32>, tensor<2x14x14x256xf32>) -> tensor<2x14x14x256xf32>
    return %5 : tensor<2x14x14x256xf32>
  }
}

// after --iree-mhlo-to-mhlo-preprocessing

builtin.module  {
  builtin.func @batch_norm_inference(%arg0: tensor<2x14x14x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>, %arg3: tensor<256xf32>, %arg4: tensor<256xf32>) -> tensor<2x14x14x256xf32> {
    %0 = mhlo.constant dense<1.000000e-03> : tensor<256xf32>
    %1 = mhlo.add %arg4, %0 : tensor<256xf32>
    %2 = "mhlo.rsqrt"(%1) : (tensor<256xf32>) -> tensor<256xf32>
    %3 = mhlo.multiply %arg1, %2 : tensor<256xf32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<2x14x14x256xf32>
    %5 = mhlo.multiply %arg0, %4 : tensor<2x14x14x256xf32>
    %6 = mhlo.multiply %arg3, %3 : tensor<256xf32>
    %7 = mhlo.subtract %arg2, %6 : tensor<256xf32>
    %8 = "mhlo.broadcast_in_dim"(%7) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<2x14x14x256xf32>
    %9 = mhlo.add %5, %8 : tensor<2x14x14x256xf32>
    return %9 : tensor<2x14x14x256xf32>
  }
  builtin.func @layer_norm_inference(%arg0: tensor<2x14x14x256xf32>, %arg1: tensor<256xf32>, %arg2: tensor<256xf32>) -> tensor<2x14x14x256xf32> {
    %0 = mhlo.constant dense<1.000000e-03> : tensor<392xf32>
    %1 = mhlo.constant dense<3.906250e-03> : tensor<392xf32>
    %2 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %3 = "mhlo.reshape"(%arg0) : (tensor<2x14x14x256xf32>) -> tensor<392x256xf32>
    %4 = "mhlo.reduce"(%3, %2) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %25 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%25) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<392x256xf32>, tensor<f32>) -> tensor<392xf32>
    %5 = "mhlo.reshape"(%arg0) : (tensor<2x14x14x256xf32>) -> tensor<392x256xf32>
    %6 = mhlo.multiply %5, %5 : tensor<392x256xf32>
    %7 = "mhlo.reduce"(%6, %2) ( {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
      %25 = mhlo.add %arg3, %arg4 : tensor<f32>
      "mhlo.return"(%25) : (tensor<f32>) -> ()
    }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<392x256xf32>, tensor<f32>) -> tensor<392xf32>
    %8 = mhlo.multiply %7, %1 : tensor<392xf32>
    %9 = mhlo.multiply %4, %1 : tensor<392xf32>
    %10 = mhlo.multiply %9, %9 : tensor<392xf32>
    %11 = mhlo.subtract %8, %10 : tensor<392xf32>
    %12 = mhlo.add %11, %0 : tensor<392xf32>
    %13 = "mhlo.rsqrt"(%12) : (tensor<392xf32>) -> tensor<392xf32>
    %14 = "mhlo.reshape"(%arg0) : (tensor<2x14x14x256xf32>) -> tensor<392x256xf32>
    %15 = mhlo.multiply %4, %1 : tensor<392xf32>
    %16 = "mhlo.broadcast_in_dim"(%15) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<392xf32>) -> tensor<392x256xf32>
    %17 = mhlo.subtract %14, %16 : tensor<392x256xf32>
    %18 = "mhlo.broadcast_in_dim"(%13) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<392xf32>) -> tensor<392x256xf32>
    %19 = mhlo.multiply %17, %18 : tensor<392x256xf32>
    %20 = "mhlo.reshape"(%19) : (tensor<392x256xf32>) -> tensor<2x14x14x256xf32>
    %21 = "mhlo.broadcast_in_dim"(%arg1) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<2x14x14x256xf32>
    %22 = mhlo.multiply %20, %21 : tensor<2x14x14x256xf32>
    %23 = "mhlo.broadcast_in_dim"(%arg2) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<256xf32>) -> tensor<2x14x14x256xf32>
    %24 = mhlo.add %22, %23 : tensor<2x14x14x256xf32>
    return %24 : tensor<2x14x14x256xf32>
  }
}