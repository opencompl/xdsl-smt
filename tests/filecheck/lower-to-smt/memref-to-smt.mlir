// RUN: xdsl-smt "%s" -p=lower-to-smt | filecheck "%s"

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

// CHECK-NEXT: builtin.module {
// CHECK-NEXT:   %0 = "smt.define_fun"() ({
// CHECK-NEXT:   ^0(%1 : !effect.state):
// CHECK-NEXT:     %2 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<126: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %3, %4 = mem_effect.alloc %1, %2
// CHECK-NEXT:     %5 = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool
// CHECK-NEXT:     %mem = "smt.utils.pair"(%4, %5) : (!mem_effect.ptr, !smt.bool) -> !smt.utils.pair<!mem_effect.ptr, !smt.bool>
// CHECK-NEXT:     %6 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<5: 8>} : () -> !smt.bv.bv<8>
// CHECK-NEXT:     %7 = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool
// CHECK-NEXT:     %value = "smt.utils.pair"(%6, %7) : (!smt.bv.bv<8>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>
// CHECK-NEXT:     %8 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<2: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %9 = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool
// CHECK-NEXT:     %i = "smt.utils.pair"(%8, %9) : (!smt.bv.bv<64>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<64>, !smt.bool>
// CHECK-NEXT:     %10 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<13: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %11 = "smt.constant_bool"() {"value" = #smt.bool_attr<false>} : () -> !smt.bool
// CHECK-NEXT:     %j = "smt.utils.pair"(%10, %11) : (!smt.bv.bv<64>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<64>, !smt.bool>
// CHECK-NEXT:     %12 = "smt.utils.first"(%value) : (!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>) -> !smt.bv.bv<8>
// CHECK-NEXT:     %13 = "smt.utils.second"(%value) : (!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:     %14 = "smt.utils.first"(%mem) : (!smt.utils.pair<!mem_effect.ptr, !smt.bool>) -> !mem_effect.ptr
// CHECK-NEXT:     %15 = "smt.utils.second"(%mem) : (!smt.utils.pair<!mem_effect.ptr, !smt.bool>) -> !smt.bool
// CHECK-NEXT:     %16 = "smt.utils.first"(%i) : (!smt.utils.pair<!smt.bv.bv<64>, !smt.bool>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %17 = "smt.utils.second"(%i) : (!smt.utils.pair<!smt.bv.bv<64>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:     %18 = "smt.utils.first"(%j) : (!smt.utils.pair<!smt.bv.bv<64>, !smt.bool>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %19 = "smt.utils.second"(%j) : (!smt.utils.pair<!smt.bv.bv<64>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:     %20 = "smt.or"(%13, %15) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:     %21 = "smt.or"(%20, %17) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:     %22 = "smt.or"(%21, %19) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:     %23 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %24 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<3: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %25 = "smt.bv.mul"(%23, %24) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %26 = "smt.bv.add"(%23, %16) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %27 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<42: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %28 = "smt.bv.mul"(%26, %27) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %29 = "smt.bv.add"(%26, %18) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %30 = mem_effect.offset_ptr %14[%29]
// CHECK-NEXT:     %31 = mem_effect.write %12, %3[%30] : !smt.bv.bv<8>
// CHECK-NEXT:     %32 = ub_effect.trigger %3
// CHECK-NEXT:     %33 = "smt.ite"(%22, %32, %31) : (!smt.bool, !effect.state, !effect.state) -> !effect.state
// CHECK-NEXT:     %34 = "smt.utils.first"(%mem) : (!smt.utils.pair<!mem_effect.ptr, !smt.bool>) -> !mem_effect.ptr
// CHECK-NEXT:     %35 = "smt.utils.second"(%mem) : (!smt.utils.pair<!mem_effect.ptr, !smt.bool>) -> !smt.bool
// CHECK-NEXT:     %36 = "smt.utils.first"(%i) : (!smt.utils.pair<!smt.bv.bv<64>, !smt.bool>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %37 = "smt.utils.second"(%i) : (!smt.utils.pair<!smt.bv.bv<64>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:     %38 = "smt.utils.first"(%j) : (!smt.utils.pair<!smt.bv.bv<64>, !smt.bool>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %39 = "smt.utils.second"(%j) : (!smt.utils.pair<!smt.bv.bv<64>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:     %40 = "smt.or"(%35, %37) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:     %41 = "smt.or"(%40, %39) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:     %42 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %43 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<3: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %44 = "smt.bv.mul"(%42, %43) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %45 = "smt.bv.add"(%42, %36) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %46 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<42: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:     %47 = "smt.bv.mul"(%45, %46) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %48 = "smt.bv.add"(%45, %38) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:     %49 = mem_effect.offset_ptr %34[%48]
// CHECK-NEXT:     %50, %val = mem_effect.read %31[%49] : !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>
// CHECK-NEXT:     %51 = ub_effect.trigger %31
// CHECK-NEXT:     %52 = "smt.ite"(%41, %51, %50) : (!smt.bool, !effect.state, !effect.state) -> !effect.state
// CHECK-NEXT:     %53 = "smt.utils.pair"(%val, %52) : (!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>, !effect.state) -> !smt.utils.pair<!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>, !effect.state>
// CHECK-NEXT:     "smt.return"(%53) : (!smt.utils.pair<!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>, !effect.state>) -> ()
// CHECK-NEXT:   }) {"fun_name" = "add"} : () -> ((!effect.state) -> !smt.utils.pair<!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>, !effect.state>)
// CHECK-NEXT: }
