//https://github.com/llvm/Polygeist/blob/main/test/polygeist-opt/affraise3.mlir

module {
  func.func @slt(%arg0: index) {
    affine.for %arg1 = 0 to 10 {
      %c = arith.cmpi slt, %arg1, %arg0 : index
      scf.if %c {
        "test.run"(%arg1) : (index) -> ()
      }
    }
    return
  }
  func.func @sle(%arg0: index) {
    affine.for %arg1 = 0 to 10 {
      %c = arith.cmpi sle, %arg1, %arg0 : index
      scf.if %c {
        "test.run"(%arg1) : (index) -> ()
      }
    }
    return
  }
  func.func @sgt(%arg0: index) {
    affine.for %arg1 = 0 to 10 {
      %c = arith.cmpi sgt, %arg1, %arg0 : index
      scf.if %c {
        "test.run"(%arg1) : (index) -> ()
      }
    }
    return
  }
  func.func @sge(%arg0: index) {
    affine.for %arg1 = 0 to 10 {
      %c = arith.cmpi sge, %arg1, %arg0 : index
      scf.if %c {
        "test.run"(%arg1) : (index) -> ()
      }
    }
    return
  }
}
