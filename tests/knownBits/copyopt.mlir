//https://github.com/llvm/Polygeist/blob/main/test/polygeist-opt/copyopt.mlir

module {
  func.func @cpy(%46: i64, %66: memref<?xi32>, %51: memref<?xi32>) {
    %c4_i64 = arith.constant 4 : i64
    %false = arith.constant false
      %47 = arith.muli %46, %c4_i64 : i64
      %48 = arith.trunci %47 : i64 to i32
      %67 = "polygeist.memref2pointer"(%66) : (memref<?xi32>) -> !llvm.ptr<i8>
      %68 = "polygeist.memref2pointer"(%51) : (memref<?xi32>) -> !llvm.ptr<i8>
      %69 = arith.extsi %48 : i32 to i64
      "llvm.intr.memcpy"(%67, %68, %69, %false) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
    return
  }
}
