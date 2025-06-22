// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize,dce | filecheck "%s"

builtin.module {
    func.func private @add() -> i8 {
        %mem = memref.alloc() : memref<3x42xi8>
        %value = arith.constant 5 : i8
        %i = arith.constant 2 : index
        %j = arith.constant 13 : index
        memref.store %value, %mem[%i, %j] : memref<3x42xi8>
        %val = memref.load %mem[%i, %j] : memref<3x42xi8>
        func.return %val : i8
    }
}

// CHECK:         %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%1 : !effect.state):
// CHECK-NEXT:      %2 = smt.bv.constant #smt.bv<126> : !smt.bv<64>
// CHECK-NEXT:      %3, %4 = mem_effect.alloc %1, %2
// CHECK-NEXT:      %5 = smt.bv.constant #smt.bv<5> : !smt.bv<8>
// CHECK-NEXT:      %6 = "smt.constant"() <{value = false}> : () -> !smt.bool
// CHECK-NEXT:      %value = "smt.utils.pair"(%5, %6) : (!smt.bv<8>, !smt.bool) -> !smt.utils.pair<!smt.bv<8>, !smt.bool>
// CHECK-NEXT:      %7 = smt.bv.constant #smt.bv<97> : !smt.bv<64>
// CHECK-NEXT:      %8 = mem_effect.offset_ptr %4[%7]
// CHECK-NEXT:      %9 = mem_effect.write %value, %3[%8] : !smt.utils.pair<!smt.bv<8>, !smt.bool>
// CHECK-NEXT:      %10 = smt.bv.constant #smt.bv<97> : !smt.bv<64>
// CHECK-NEXT:      %11 = mem_effect.offset_ptr %4[%10]
// CHECK-NEXT:      %12, %val = mem_effect.read %9[%11] : !smt.utils.pair<!smt.bv<8>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%val, %12) : (!smt.utils.pair<!smt.bv<8>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "add"} : () -> ((!effect.state) -> (!smt.utils.pair<!smt.bv<8>, !smt.bool>, !effect.state))
