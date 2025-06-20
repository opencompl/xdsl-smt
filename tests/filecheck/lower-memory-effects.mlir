// RUN: xdsl-smt %s -p=lower-memory-effects,lower-pairs --split-input-file | filecheck %s

%state = "smt.declare_const"() : () -> !effect.state
%size = "smt.declare_const"() : () -> !smt.bv<64>
%state2, %ptr = mem_effect.alloc %state, %size
"test.op"(%state2, %ptr) : (!effect.state, !mem_effect.ptr) -> ()

// CHECK:         %state = "smt.declare_const"() : () -> !effect.state
// CHECK-NEXT:    %size = "smt.declare_const"() : () -> !smt.bv<64>
// CHECK-NEXT:    %memory = memory.get_memory %state
// CHECK-NEXT:    %bid, %0 = memory.get_fresh_block_id %memory
// CHECK-NEXT:    %block = memory.get_block %0[%bid]
// CHECK-NEXT:    %1 = "smt.constant"() <{value = true}> : () -> !smt.bool
// CHECK-NEXT:    %block_1 = memory.set_block_live_marker %block, %1
// CHECK-NEXT:    %block_2 = memory.set_block_size %block_1, %size
// CHECK-NEXT:    %memory_1 = memory.set_block %block_2, %0[%bid]
// CHECK-NEXT:    %2 = "smt.bv.constant"() {value = #smt.bv<0> : !smt.bv<64>} : () -> !smt.bv<64>
// CHECK-NEXT:    %ptr = "smt.utils.pair"(%bid, %2) : (!memory.block_id, !smt.bv<64>) -> !smt.utils.pair<!memory.block_id, !smt.bv<64>>
// CHECK-NEXT:    %state_1 = memory.set_memory %state, %memory_1
// CHECK-NEXT:    "test.op"(%state_1, %ptr) : (!effect.state, !smt.utils.pair<!memory.block_id, !smt.bv<64>>) -> ()

// -----

%ptr = "smt.declare_const"() : () -> !mem_effect.ptr
%offset = "smt.declare_const"() : () -> !smt.bv<64>
%ptr2 = mem_effect.offset_ptr %ptr[%offset]
"test.op"(%ptr2) : (!mem_effect.ptr) -> ()

// CHECK:         %ptr_first = "smt.declare_const"() : () -> !memory.block_id
// CHECK-NEXT:    %ptr_second = "smt.declare_const"() : () -> !smt.bv<64>
// CHECK-NEXT:    %offset = "smt.declare_const"() : () -> !smt.bv<64>
// CHECK-NEXT:    %ptr_offset = "smt.bv.add"(%ptr_second, %offset) : (!smt.bv<64>, !smt.bv<64>) -> !smt.bv<64>
// CHECK-NEXT:    %ptr = "smt.utils.pair"(%ptr_first, %ptr_offset) : (!memory.block_id, !smt.bv<64>) -> !smt.utils.pair<!memory.block_id, !smt.bv<64>>
// CHECK-NEXT:    "test.op"(%ptr) : (!smt.utils.pair<!memory.block_id, !smt.bv<64>>) -> ()

// -----

%state = "smt.declare_const"() : () -> !effect.state
%ptr = "smt.declare_const"() : () -> !mem_effect.ptr
%state2, %value = mem_effect.read %state[%ptr] : !smt.utils.pair<!smt.bv<8>, !smt.bool>
"test.op"(%state2, %value) : (!effect.state, !smt.utils.pair<!smt.bv<8>, !smt.bool>) -> ()

// CHECK:         %state = "smt.declare_const"() : () -> !effect.state
// CHECK-NEXT:    %ptr_first = "smt.declare_const"() : () -> !memory.block_id
// CHECK-NEXT:    %ptr_second = "smt.declare_const"() : () -> !smt.bv<64>
// CHECK-NEXT:    %memory = memory.get_memory %state
// CHECK-NEXT:    %block = memory.get_block %memory[%ptr_first]
// CHECK-NEXT:    %block_bytes = memory.get_block_bytes %block
// CHECK-NEXT:    %block_size = memory.get_block_size %block
// CHECK-NEXT:    %0 = "smt.bv.add"(%ptr_second, %block_size) : (!smt.bv<64>, !smt.bv<64>) -> !smt.bv<64>
// CHECK-NEXT:    %offset_in_bounds = "smt.bv.ule"(%0, %block_size) : (!smt.bv<64>, !smt.bv<64>) -> !smt.bool
// CHECK-NEXT:    %offset_not_in_bounds = "smt.not"(%offset_in_bounds) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    %read = memory.read_bytes %block_bytes[%ptr_second] : !smt.utils.pair<!smt.bv<8>, !smt.bool>
// CHECK-NEXT:    %state_1 = memory.set_memory %state, %memory
// CHECK-NEXT:    %1 = ub_effect.trigger %state
// CHECK-NEXT:    %state2 = "smt.ite"(%offset_not_in_bounds, %1, %state_1) : (!smt.bool, !effect.state, !effect.state) -> !effect.state
// CHECK-NEXT:    "test.op"(%state2, %read) : (!effect.state, !smt.utils.pair<!smt.bv<8>, !smt.bool>) -> ()

// -----

%state = "smt.declare_const"() : () -> !effect.state
%ptr = "smt.declare_const"() : () -> !mem_effect.ptr
%value = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bv<16>, !smt.bool>
%state2 = mem_effect.write %value, %state[%ptr] : !smt.utils.pair<!smt.bv<16>, !smt.bool>
"test.op"(%state2) : (!effect.state) -> ()

// CHECK:         %state = "smt.declare_const"() : () -> !effect.state
// CHECK-NEXT:    %ptr_first = "smt.declare_const"() : () -> !memory.block_id
// CHECK-NEXT:    %ptr_second = "smt.declare_const"() : () -> !smt.bv<64>
// CHECK-NEXT:    %value_first = "smt.declare_const"() : () -> !smt.bv<16>
// CHECK-NEXT:    %value_second = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT:    %value = "smt.utils.pair"(%value_first, %value_second) : (!smt.bv<16>, !smt.bool) -> !smt.utils.pair<!smt.bv<16>, !smt.bool>
// CHECK-NEXT:    %memory = memory.get_memory %state
// CHECK-NEXT:    %block = memory.get_block %memory[%ptr_first]
// CHECK-NEXT:    %block_bytes = memory.get_block_bytes %block
// CHECK-NEXT:    %block_size = memory.get_block_size %block
// CHECK-NEXT:    %0 = "smt.bv.add"(%ptr_second, %block_size) : (!smt.bv<64>, !smt.bv<64>) -> !smt.bv<64>
// CHECK-NEXT:    %offset_in_bounds = "smt.bv.ule"(%0, %block_size) : (!smt.bv<64>, !smt.bv<64>) -> !smt.bool
// CHECK-NEXT:    %offset_not_in_bounds = "smt.not"(%offset_in_bounds) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:    %bytes = memory.write_bytes %value, %block_bytes[%ptr_second] : !smt.utils.pair<!smt.bv<16>, !smt.bool>
// CHECK-NEXT:    %block_1 = memory.set_block_bytes %block, %bytes
// CHECK-NEXT:    %memory_1 = memory.set_block %block_1, %memory[%ptr_first]
// CHECK-NEXT:    %state_1 = memory.set_memory %state, %memory_1
// CHECK-NEXT:    %1 = ub_effect.trigger %state
// CHECK-NEXT:    %state2 = "smt.ite"(%offset_not_in_bounds, %1, %state_1) : (!smt.bool, !effect.state, !effect.state) -> !effect.state
// CHECK-NEXT:    "test.op"(%state2) : (!effect.state) -> ()
