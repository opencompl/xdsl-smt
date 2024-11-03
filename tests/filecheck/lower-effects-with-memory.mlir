// RUN: xdsl-smt %s -p=lower-effects-with-memory --split-input-file | filecheck %s

%state = "smt.declare_const"() : () -> !effect.state
%size = "smt.bv.constant"() {value = #smt.bv.bv_val<32 : 64>} : () -> !smt.bv.bv<64>
%state2, %ptr = mem_effect.alloc %state, %size

// CHECK:         %state_first = "smt.declare_const"() : () -> !memory.memory
// CHECK-NEXT:    %state_second = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:    %size = "smt.bv.constant"() {"value" = #smt.bv.bv_val<32: 64>} : () -> !smt.bv.bv<64>
// CHECK-NEXT:    %bid = memory.get_fresh_block_id %state_first
// CHECK-NEXT:    %block = memory.get_block %state_first[%bid]
// CHECK-NEXT:    %0 = "smt.constant_bool"() {"value" = #smt.bool_attr<true>} : () -> !smt.bool
// CHECK-NEXT:    %block_1 = memory.set_block_live_marker %block, %0
// CHECK-NEXT:    %block_2 = memory.set_block_size %block_1, %size
// CHECK-NEXT:    %memory = memory.set_block %state_first[%bid], %block_2
