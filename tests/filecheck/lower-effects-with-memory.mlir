// RUN: xdsl-smt %s -p=lower-effects-with-memory,lower-pairs --split-input-file | filecheck %s

%state = "smt.declare_const"() : () -> !effect.state
%size = "smt.declare_const"() : () -> !smt.bv.bv<64>
%state2, %ptr = mem_effect.alloc %state, %size

// CHECK:         %state_first = "smt.declare_const"() : () -> !memory.memory
// CHECK-NEXT:    %state_second = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:    %size = "smt.declare_const"() : () -> !smt.bv.bv<64>
// CHECK-NEXT:    %bid = memory.get_fresh_block_id %state_first
// CHECK-NEXT:    %block = memory.get_block %state_first[%bid]
// CHECK-NEXT:    %0 = "smt.constant_bool"() {"value" = #smt.bool_attr<true>} : () -> !smt.bool
// CHECK-NEXT:    %block_1 = memory.set_block_live_marker %block, %0
// CHECK-NEXT:    %block_2 = memory.set_block_size %block_1, %size
// CHECK-NEXT:    %memory = memory.set_block %state_first[%bid], %block_2

// -----

%ptr = "smt.declare_const"() : () -> !mem_effect.ptr
%offset = "smt.declare_const"() : () -> !smt.bv.bv<64>
%ptr2 = mem_effect.offset_ptr %ptr[%offset]
"test.op"(%ptr2) : (!mem_effect.ptr) -> ()

// CHECK:         %ptr_first = "smt.declare_const"() : () -> !memory.block_id
// CHECK-NEXT:    %ptr_second = "smt.declare_const"() : () -> !smt.bv.bv<64>
// CHECK-NEXT:    %offset = "smt.declare_const"() : () -> !smt.bv.bv<64>
// CHECK-NEXT:    %ptr_offset = "smt.bv.add"(%ptr_second, %offset) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:    %ptr = "smt.utils.pair"(%ptr_first, %ptr_offset) : (!memory.block_id, !smt.bv.bv<64>) -> !smt.utils.pair<!memory.block_id, !smt.bv.bv<64>>
// CHECK-NEXT:    "test.op"(%ptr) : (!smt.utils.pair<!memory.block_id, !smt.bv.bv<64>>) -> ()
