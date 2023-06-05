//https://github.com/llvm/Polygeist/blob/main/test/polygeist-opt/affinecfg.mlir

module {
  func.func @_Z7runTestiPPc(%arg0: index, %arg2: memref<?xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %1 = arith.addi %arg0, %c1 : index
    affine.for %arg3 = 0 to 2 {
      %2 = arith.muli %arg3, %1 : index
      affine.for %arg4 = 0 to 2 {
        %3 = arith.addi %2, %arg4 : index
        memref.store %c0_i32, %arg2[%3] : memref<?xi32>
      }
    }
    return
  }

}

module {
func.func @kernel_nussinov(%arg0: i32, %arg2: memref<i32>) {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  %c1_i32 = arith.constant 1 : i32
  %c59 = arith.constant 59 : index
  %c100_i32 = arith.constant 100 : i32
  affine.for %arg3 = 0 to 60 {
    %0 = arith.subi %c59, %arg3 : index
    %1 = arith.index_cast %0 : index to i32
    %2 = arith.cmpi slt, %1, %c100_i32 : i32
    scf.if %2 {
      affine.store %arg0, %arg2[] : memref<i32>
    }
  }
  return
}
}

module {
  func.func private @run()

  func.func @minif(%arg4: i32, %arg5 : i32, %arg10 : index) {
    %c0_i32 = arith.constant 0 : i32

    affine.for %i = 0 to 10 {
      %70 = arith.index_cast %arg10 : index to i32
      %71 = arith.muli %70, %arg5 : i32
      %73 = arith.divui %71, %arg5 : i32
      %75 = arith.muli %73, %arg5 : i32
      %79 = arith.subi %arg4, %75 : i32
      %81 = arith.cmpi sle, %arg5, %79 : i32
      %83 = arith.select %81, %arg5, %79 : i32
      %92 = arith.cmpi slt, %c0_i32, %83 : i32
      scf.if %92 {
        func.call @run() : () -> ()
        scf.yield
      }
    }
    return
  }
}

module {
  llvm.func @atoi(!llvm.ptr<i8>) -> i32
func.func @_Z7runTestiPPc(%arg0: i32, %39: memref<?xi32>, %arg1: !llvm.ptr<i8>) attributes {llvm.linkage = #llvm.linkage<external>} {
  %c2_i32 = arith.constant 2 : i32
  %c16_i32 = arith.constant 16 : i32
    %58 = llvm.call @atoi(%arg1) : (!llvm.ptr<i8>) -> i32
  %40 = arith.divsi %58, %c16_i32 : i32
  affine.for %arg2 = 1 to 10 {
      %62 = arith.index_cast %arg2 : index to i32
      %67 = arith.muli %58, %62 : i32
      %69 = arith.addi %67, %40 : i32
        %75 = arith.addi %69, %58 : i32
        %76 = arith.index_cast %75 : i32 to index
        memref.store %c2_i32, %39[%76] : memref<?xi32>
  }
  return
}
}

module {
  func.func @c(%71: memref<?xf32>, %39: i64) {
      affine.parallel (%arg2, %arg3) = (0, 0) to (42, 512) {
        %262 = arith.index_cast %arg2 : index to i32
        %a264 = arith.extsi %262 : i32 to i64
        %268 = arith.cmpi slt, %a264, %39 : i64
        scf.if %268 {
          "test.something"() : () -> ()
        }
      }
    return
  }
}
