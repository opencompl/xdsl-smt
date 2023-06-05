//https://github.com/llvm/Polygeist/blob/main/test/polygeist-opt/affraise2.mlir

module {
  func.func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %19 : memref<?xf32> ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.if %12 {
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.muli %15, %c4 : index
      %17 = arith.divui %16, %c4 : index
      scf.for %arg2 = %c0 to %17 step %c1 {
        %20 = memref.load %19[%arg2] : memref<?xf32>
        memref.store %20, %18[%arg2] : memref<?xf32>
      }
    }
    return
  }
}
