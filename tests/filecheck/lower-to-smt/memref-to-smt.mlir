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

// CHECK:      builtin.module {
// CHECK-NEXT:   %0 = "smt.define_fun"() ({
// CHECK-NEXT:   ^0(%1 : !effect.state):
// CHECK-NEXT:     %2 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<126: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %3, %4 = mem_effect.alloc %1, %2
// CHECK-NEXT:     %5 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<5: 8>} : () -> !smt.bv.bv<8>
// CHECK-NEXT:     %6 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<2: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %7 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<13: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %8 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %9 = "smt.bv.add"(%8, %6) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %10 = "smt.bv.add"(%9, %7) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %11 = mem_effect.offset_ptr %4[%10]
// CHECK-NEXT:     %12 = mem_effect.write %5, %3[%11] : !smt.bv.bv<8>
// CHECK-NEXT:     %13 = ub_effect.trigger %3
// CHECK-NEXT:     %14 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %15 = "smt.bv.sge"(%6, %14) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bool
// CHECK-NEXT:     %16 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<3: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %17 = "smt.bv.slt"(%6, %16) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bool
// CHECK-NEXT:     %18 = "smt.and"(%15, %17) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:     %19 = "smt.bv.sge"(%7, %14) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bool
// CHECK-NEXT:     %20 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<42: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %21 = "smt.bv.slt"(%7, %20) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bool
// CHECK-NEXT:     %22 = "smt.and"(%19, %21) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:     %23 = "smt.and"(%18, %22) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:     %24 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %25 = "smt.bv.add"(%24, %6) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %26 = "smt.bv.add"(%25, %7) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %27 = mem_effect.offset_ptr %4[%26]
// CHECK-NEXT:     %28, %val = mem_effect.read %12[%27] : !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>
// CHECK-NEXT:     %29 = ub_effect.trigger %12
// CHECK-NEXT:     %30 = "smt.not"(%23) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:     %31 = "smt.ite"(%30, %29, %28) : (!smt.bool, !effect.state, !effect.state) -> !effect.state
// CHECK-NEXT:     %32 = "smt.utils.pair"(%val, %31) : (!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>, !effect.state) -> !smt.utils.pair<!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>, !effect.state>
// CHECK-NEXT:     "smt.return"(%32) : (!smt.utils.pair<!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>, !effect.state>) -> ()
// CHECK-NEXT:   }) {"fun_name" = "add"} : () -> ((!effect.state) -> !smt.utils.pair<!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>, !effect.state>)
// CHECK-NEXT: }
