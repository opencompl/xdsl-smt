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
// CHECK-NEXT:    %memory = memory.set_block %block_2, %state_first[%bid]

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

// -----

%state = "smt.declare_const"() : () -> !effect.state
%ptr = "smt.declare_const"() : () -> !mem_effect.ptr
%state2, %value = mem_effect.read %state[%ptr] : !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>
"test.op"(%state2, %value) : (!effect.state, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>) -> ()

// CHECK:         %state_first = "smt.declare_const"() : () -> !memory.memory
// CHECK-NEXT:    %state_second = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:    %ptr_first = "smt.declare_const"() : () -> !memory.block_id
// CHECK-NEXT:    %ptr_second = "smt.declare_const"() : () -> !smt.bv.bv<64>
// CHECK-NEXT:    %block = memory.get_block %state_first[%ptr_first]
// CHECK-NEXT:    %block_bytes = memory.get_block_bytes %block
// CHECK-NEXT:    %block_size = memory.get_block_size %block
// CHECK-NEXT:    %0 = "smt.bv.add"(%ptr_second, %block_size) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:    %offset_in_bounds = "smt.bv.ule"(%0, %block_size) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bool
// CHECK-NEXT:    %1 = "smt.or"(%state_second, %offset_in_bounds) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %read = memory.read_bytes %block_bytes[%ptr_second] : !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>
// CHECK-NEXT:    %state = "smt.utils.pair"(%state_first, %1) : (!memory.memory, !smt.bool) -> !smt.utils.pair<!memory.memory, !smt.bool>
// CHECK-NEXT:    "test.op"(%state, %read) : (!smt.utils.pair<!memory.memory, !smt.bool>, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>) -> ()

// -----

%state = "smt.declare_const"() : () -> !effect.state
%ptr = "smt.declare_const"() : () -> !mem_effect.ptr
%value = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bv.bv<16>, !smt.bool>
%state2 = mem_effect.write %value, %state[%ptr] : !smt.utils.pair<!smt.bv.bv<16>, !smt.bool>
"test.op"(%state2) : (!effect.state) -> ()

// CHECK:         %state_first = "smt.declare_const"() : () -> !memory.memory
// CHECK-NEXT:    %state_second = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:    %ptr_first = "smt.declare_const"() : () -> !memory.block_id
// CHECK-NEXT:    %ptr_second = "smt.declare_const"() : () -> !smt.bv.bv<64>
// CHECK-NEXT:    %value_first = "smt.declare_const"() : () -> !smt.bv.bv<16>
// CHECK-NEXT:    %value_second = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:    %value = "smt.utils.pair"(%value_first, %value_second) : (!smt.bv.bv<16>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<16>, !smt.bool>
// CHECK-NEXT:    %block = memory.get_block %state_first[%ptr_first]
// CHECK-NEXT:    %block_bytes = memory.get_block_bytes %block
// CHECK-NEXT:    %block_size = memory.get_block_size %block
// CHECK-NEXT:    %0 = "smt.bv.add"(%ptr_second, %block_size) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bv.bv<64>
// CHECK-NEXT:    %offset_in_bounds = "smt.bv.ule"(%0, %block_size) : (!smt.bv.bv<64>, !smt.bv.bv<64>) -> !smt.bool
// CHECK-NEXT:    %1 = "smt.or"(%state_second, %offset_in_bounds) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:    %bytes = memory.write_bytes %value, %block_bytes[%ptr_second] : !smt.utils.pair<!smt.bv.bv<16>, !smt.bool>
// CHECK-NEXT:    %block_1 = memory.set_block_bytes %block_bytes, %block
// CHECK-NEXT:    %memory = memory.set_block %block_1, %state_first[%ptr_first]
// CHECK-NEXT:    %state = "smt.utils.pair"(%memory, %1) : (!memory.memory, !smt.bool) -> !smt.utils.pair<!memory.memory, !smt.bool>
// CHECK-NEXT:    "test.op"(%state) : (!smt.utils.pair<!memory.memory, !smt.bool>) -> ()
