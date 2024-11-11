// RUN: xdsl-tv %s %S/write-ub.mlir.output | z3 -in | filecheck %s

// This file uses `write-ub.mlir.output`

builtin.module {
    func.func private @foo(%mem: memref<3x42xi8>) {
        %value = arith.constant 5 : i8
        %i = arith.constant 4 : index
        %j = arith.constant 13 : index
        // UB: Write out of bounds
        memref.store %value, %mem[%i, %j] : memref<3x42xi8>
        func.return
    }
}

// CHECK: unsat
