//https://github.com/llvm/Polygeist/blob/main/test/polygeist-opt/affparmerge.mlir

module {
  func.func @f(%636: index,  %603: memref<?xf64>) {
    %c512_i32 = arith.constant 512 : i32
        affine.parallel (%arg7, %arg8) = (0, 0) to (symbol(%636), 512) {
          %706 = arith.index_cast %arg7 : index to i32
          %707 = arith.muli %706, %c512_i32 : i32
            %708 = arith.index_cast %arg8 : index to i32
            %709 = arith.addi %707, %708 : i32
              %712 = arith.sitofp %709 : i32 to f64
              affine.store %712, %603[%arg8 + %arg7 * 512] : memref<?xf64>
        }
    return
  }

}
