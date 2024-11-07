// RUN: xdsl-tv %s %S/remove-alloc.mlir.output | z3 -in | filecheck %s

// This file uses `remove-alloc.mlir.output`

builtin.module {
    func.func private @foo() -> i8 {
        %mem = memref.alloc() : memref<3x42xi8>
        %value = arith.constant 5 : i8
        %i = arith.constant 2 : index
        %j = arith.constant 13 : index
        memref.store %value, %mem[%i, %j] : memref<3x42xi8>
        %val = memref.load %mem[%i, %j] : memref<3x42xi8>
        func.return %val : i8
    }
}

// CHECK: unsat
