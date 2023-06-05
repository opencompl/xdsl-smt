//https://github.com/iml130/mlir-emitc/blob/main/test/Conversion/arith-to-emitc.mlir

func.func @arith_index_cast(%arg0: tensor<index>, %arg1: tensor<2xi32>, %arg2: tensor<2x2xi32>) -> tensor<2xindex> {
  %0 = "arith.index_cast"(%arg0) : ( tensor<index>) -> tensor<i32>
  %1 = "arith.index_cast"(%arg1) : (tensor<2xi32>) -> tensor<2xindex>
  %2 = "arith.index_cast"(%arg2) : (tensor<2x2xi32>) -> tensor<2x2xindex>
  return %1 : tensor<2xindex>
}
