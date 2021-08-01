func @add_dynamic(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>) -> tensor<?xf32> {
 %0 = "mhlo.add"(%lhs, %rhs) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
 return %0 : tensor<?xf32>
}