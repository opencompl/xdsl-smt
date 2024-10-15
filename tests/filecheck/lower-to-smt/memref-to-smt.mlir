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

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%1 : !effect.state):
// CHECK-NEXT:      %2 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<126: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:      %3, %4 = mem_effect.alloc %1, %2
// CHECK-NEXT:      %5 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<5: 8>} : () -> !smt.bv.bv<8>
// CHECK-NEXT:      %6 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<97: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:      %7 = mem_effect.offset_ptr %4[%6]
// CHECK-NEXT:      %8 = mem_effect.write %5, %3[%7] : !smt.bv.bv<8>
// CHECK-NEXT:      %9, %val = mem_effect.read %8[%7] : !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>
// CHECK-NEXT:      %10 = "smt.utils.pair"(%val, %9) : (!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>, !effect.state) -> !smt.utils.pair<!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>, !effect.state>
// CHECK-NEXT:      "smt.return"(%10) : (!smt.utils.pair<!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>, !effect.state>) -> ()
// CHECK-NEXT:    }) {"fun_name" = "add"} : () -> ((!effect.state) -> !smt.utils.pair<!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>, !effect.state>)
// CHECK-NEXT:  }
