//https://github.com/llvm/Polygeist/blob/main/test/polygeist-opt/affifcombine.mlir

#set0 = affine_set<(d0, d1) : (d0 + d1 * 512 == 0)>

module {
  func.func private @use(index)
  func.func @k(%636: index,  %603: memref<?xf64>) {
    %c512_i32 = arith.constant 512 : i32
    affine.parallel (%arg7) = (0) to (symbol(%636)) {
      %706 = arith.index_cast %arg7 : index to i32
      %707 = arith.muli %706, %c512_i32 : i32
      affine.parallel (%arg8) = (0) to (512) {
        %708 = arith.index_cast %arg8 : index to i32
        %709 = arith.addi %707, %708 : i32
        %ifres = affine.if #set0(%arg8, %arg7) -> f64 {
          %712 = arith.sitofp %709 : i32 to f64
          func.call @use(%arg7) : (index) -> ()
          affine.yield %712 : f64
        } else {
          %712 = arith.sitofp %708 : i32 to f64
          func.call @use(%arg8) : (index) -> ()
          affine.yield %712 : f64
        }
        affine.if #set0(%arg8, %arg7) {
          func.call @use(%arg7) : (index) -> ()
        } else {
          func.call @use(%arg8) : (index) -> ()
        }
        affine.store %ifres, %603[0] : memref<?xf64>
      }
    }
    return
  }

  func.func @h(%636: index,  %603: memref<?xf64>) {
    %c512_i32 = arith.constant 512 : i32
    affine.parallel (%arg7) = (0) to (symbol(%636)) {
      %706 = arith.index_cast %arg7 : index to i32
      %707 = arith.muli %706, %c512_i32 : i32
      affine.parallel (%arg8) = (0) to (512) {
        %708 = arith.index_cast %arg8 : index to i32
        %709 = arith.addi %707, %708 : i32
        %ifres = affine.if #set0(%arg8, %arg7) -> f64 {
          %712 = arith.sitofp %709 : i32 to f64
          func.call @use(%arg7) : (index) -> ()
          affine.yield %712 : f64
        } else {
          %712 = arith.sitofp %708 : i32 to f64
          func.call @use(%arg8) : (index) -> ()
          affine.yield %712 : f64
        }
        affine.if #set0(%arg8, %arg7) {
          func.call @use(%arg7) : (index) -> ()
        }
        affine.store %ifres, %603[0] : memref<?xf64>
      }
    }
    return
  }

  func.func @g(%636: index,  %603: memref<?xf64>) {
    %c512_i32 = arith.constant 512 : i32

    affine.parallel (%arg7) = (0) to (symbol(%636)) {
      %706 = arith.index_cast %arg7 : index to i32
      %707 = arith.muli %706, %c512_i32 : i32
      affine.parallel (%arg8) = (0) to (512) {
        %708 = arith.index_cast %arg8 : index to i32
        %709 = arith.addi %707, %708 : i32
        affine.if #set0(%arg8, %arg7) {
          func.call @use(%arg7) : (index) -> ()
        }
        %ifres = affine.if #set0(%arg8, %arg7) -> f64 {
          %712 = arith.sitofp %709 : i32 to f64
          func.call @use(%arg7) : (index) -> ()
          affine.yield %712 : f64
        } else {
          %712 = arith.sitofp %708 : i32 to f64
          func.call @use(%arg8) : (index) -> ()
          affine.yield %712 : f64
        }
        affine.store %ifres, %603[0] : memref<?xf64>
      }
    }
    return
  }

  func.func @f(%636: index,  %603: memref<?xf64>) {
    %c512_i32 = arith.constant 512 : i32

    affine.parallel (%arg7) = (0) to (symbol(%636)) {
      %706 = arith.index_cast %arg7 : index to i32
      %707 = arith.muli %706, %c512_i32 : i32
      affine.parallel (%arg8) = (0) to (512) {
        %708 = arith.index_cast %arg8 : index to i32
        %709 = arith.addi %707, %708 : i32
        affine.if #set0(%arg8, %arg7) {
          func.call @use(%arg7) : (index) -> ()
        } else {
          func.call @use(%arg8) : (index) -> ()
        }
        %ifres = affine.if #set0(%arg8, %arg7) -> f64 {
          %712 = arith.sitofp %709 : i32 to f64
          func.call @use(%arg7) : (index) -> ()
          affine.yield %712 : f64
        } else {
          %712 = arith.sitofp %708 : i32 to f64
          func.call @use(%arg8) : (index) -> ()
          affine.yield %712 : f64
        }
        affine.store %ifres, %603[0] : memref<?xf64>
      }
    }
    return
  }
}
