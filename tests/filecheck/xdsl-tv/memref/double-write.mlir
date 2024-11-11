// RUN: xdsl-tv %s %S/double-write.mlir.output | z3 -in | filecheck %s

// This file uses `double-write.mlir.output`

builtin.module {
    func.func private @foo(%mem: memref<3x42xi8>) {
        %value = arith.constant 5 : i8
        %i = arith.constant 2 : index
        %j = arith.constant 13 : index
        memref.store %value, %mem[%i, %j] : memref<3x42xi8>
        memref.store %value, %mem[%i, %j] : memref<3x42xi8>
        func.return
    }
}

// CHECK: unsat
