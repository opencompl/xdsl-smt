// RUN: xdsl-smt "%s" | xdsl-smt | filecheck "%s"

builtin.module {
    %state = "smt.declare_const"() : () -> !effect.state
    %size = "smt.bv.constant"() {value = #smt.bv.bv_val<32 : 64>} : () -> !smt.bv<64>
    %state2, %ptr = mem_effect.alloc %state, %size
    %offset = "smt.bv.constant"() {value = #smt.bv.bv_val<8 : 64>} : () -> !smt.bv<64>
    %new_ptr = mem_effect.offset_ptr %ptr[%offset]
    %val = "smt.bv.constant"() {value = #smt.bv.bv_val<42 : 16>} : () -> !smt.bv<16>
    %poison = "smt.constant"() <{value = false}> : () -> !smt.bool
    %poisoned_val = "smt.utils.pair"(%val, %poison) : (!smt.bv<16>, !smt.bool) -> !smt.utils.pair<!smt.bv<16>, !smt.bool>
    %state3 = mem_effect.write %poisoned_val, %state2[%new_ptr] : !smt.utils.pair<!smt.bv<16>, !smt.bool>
    %state4, %val2 = mem_effect.read %state3[%new_ptr] : !smt.utils.pair<!smt.bv<16>, !smt.bool>
    mem_effect.dealloc %state4, %ptr
}

// CHECK:         %state = "smt.declare_const"() : () -> !effect.state
// CHECK-NEXT:    %size = "smt.bv.constant"() {value = #smt.bv.bv_val<32: 64>} : () -> !smt.bv<64>
// CHECK-NEXT:    %state2, %ptr = mem_effect.alloc %state, %size
// CHECK-NEXT:    %offset = "smt.bv.constant"() {value = #smt.bv.bv_val<8: 64>} : () -> !smt.bv<64>
// CHECK-NEXT:    %new_ptr = mem_effect.offset_ptr %ptr[%offset]
// CHECK-NEXT:    %val = "smt.bv.constant"() {value = #smt.bv.bv_val<42: 16>} : () -> !smt.bv<16>
// CHECK-NEXT:    %poison = "smt.constant"() <{value = false}> : () -> !smt.bool
// CHECK-NEXT:    %poisoned_val = "smt.utils.pair"(%val, %poison) : (!smt.bv<16>, !smt.bool) -> !smt.utils.pair<!smt.bv<16>, !smt.bool>
// CHECK-NEXT:    %state3 = mem_effect.write %poisoned_val, %state2[%new_ptr] : !smt.utils.pair<!smt.bv<16>, !smt.bool>
// CHECK-NEXT:    %state4, %val2 = mem_effect.read %state3[%new_ptr] : !smt.utils.pair<!smt.bv<16>, !smt.bool>
// CHECK-NEXT:    %0 = mem_effect.dealloc %state4, %ptr
