// RUN: xdsl-smt "%s" | xdsl-smt | filecheck "%s"

builtin.module {
    %state = "smt.declare_const"() : () -> !effect.state
    %size = "smt.bv.constant"() {value = #smt.bv.bv_val<32 : 64>} : () -> !smt.bv.bv<64>
    %state2, %ptr = mem_effect.alloc %state, %size
    %offset = "smt.bv.constant"() {value = #smt.bv.bv_val<8 : 64>} : () -> !smt.bv.bv<64>
    %new_ptr = mem_effect.offset_ptr %ptr[%offset]
    %val = "smt.bv.constant"() {value = #smt.bv.bv_val<42 : 16>} : () -> !smt.bv.bv<16>
    %state3 = mem_effect.write %state2[%new_ptr], %val : !smt.bv.bv<16>
    %val2 = mem_effect.read %state3[%new_ptr] : !smt.bv.bv<16>
    mem_effect.dealloc %state3, %ptr
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %state = "smt.declare_const"() : () -> !effect.state
// CHECK-NEXT:    %size = "smt.bv.constant"() {"value" = #smt.bv.bv_val<32: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:    %state2, %ptr = mem_effect.alloc %state, %size
// CHECK-NEXT:    %offset = "smt.bv.constant"() {"value" = #smt.bv.bv_val<8: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:    %new_ptr = mem_effect.offset_ptr %ptr[%offset]
// CHECK-NEXT:    %val = "smt.bv.constant"() {"value" = #smt.bv.bv_val<42: 16>} : () -> !smt.bv.bv<16>
// CHECK-NEXT:    %state3 = mem_effect.write %state2[%new_ptr], %val : !smt.bv.bv<16>
// CHECK-NEXT:    %val2 = mem_effect.read %state3[%new_ptr] : !smt.bv.bv<16>
// CHECK-NEXT:    %0 = mem_effect.dealloc %state3, %ptr
