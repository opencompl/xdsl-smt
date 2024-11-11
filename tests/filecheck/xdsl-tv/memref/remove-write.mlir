// RUN: xdsl-tv %s %S/remove-write.mlir.output | z3 -in | filecheck %s

// This file uses `remove-write.mlir.output`

builtin.module {
    func.func private @foo(%mem: memref<3x42xi8>) {
        %value = arith.constant 5 : i8
        %i = arith.constant 2 : index
        %j = arith.constant 13 : index
        memref.store %value, %mem[%i, %j] : memref<3x42xi8>
        func.return
    }
}

// CHECK-NOT: unsat
// CHECK: sat
