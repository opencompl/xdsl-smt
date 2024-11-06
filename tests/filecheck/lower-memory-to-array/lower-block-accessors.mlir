// RUN: xdsl-smt %s -p=lower-memory-to-array,lower-pairs | filecheck %s
// RUN: xdsl-smt %s -p=lower-memory-to-array,lower-pairs -t=smt | z3 -in

%block = "smt.declare_const"() : () -> !memory.block
%new_bytes = "smt.declare_const"() : () -> !memory.bytes
%new_live_marker = "smt.declare_const"() : () -> !smt.bool
%new_size = "smt.declare_const"() : () -> !smt.bv.bv<64>

%bytes = memory.get_block_bytes %block
%live_marker = memory.get_block_live_marker %block
%size = memory.get_block_size %block

%new_block1 = memory.set_block_bytes %block, %new_bytes
%new_block2 = memory.set_block_live_marker %new_block1, %new_live_marker
%new_block3 = memory.set_block_size %new_block2, %new_size

"test.op"(%bytes, %live_marker, %size, %new_block3) : (!memory.bytes, !smt.bool, !smt.bv.bv<64>, !memory.block) -> ()

// CHECK:      %block_first = "smt.declare_const"() : () -> !smt.array.array<!smt.int.int, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>>
// CHECK-NEXT: %block_second_first = "smt.declare_const"() : () -> !smt.bv.bv<64>
// CHECK-NEXT: %block_second_second = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT: %new_bytes = "smt.declare_const"() : () -> !smt.array.array<!smt.int.int, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>>
// CHECK-NEXT: %new_live_marker = "smt.declare_const"() : () -> !smt.bool
// CHECK-NEXT: %new_size = "smt.declare_const"() : () -> !smt.bv.bv<64>
// CHECK-NEXT: %0 = "smt.utils.pair"(%new_size, %new_live_marker) : (!smt.bv.bv<64>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<64>, !smt.bool>
// CHECK-NEXT: %new_block3 = "smt.utils.pair"(%new_bytes, %0) : (!smt.array.array<!smt.int.int, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>>, !smt.utils.pair<!smt.bv.bv<64>, !smt.bool>) -> !smt.utils.pair<!smt.array.array<!smt.int.int, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>>, !smt.utils.pair<!smt.bv.bv<64>, !smt.bool>>
// CHECK-NEXT: "test.op"(%block_first, %block_second_second, %block_second_first, %new_block3) : (!smt.array.array<!smt.int.int, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>>, !smt.bool, !smt.bv.bv<64>, !smt.utils.pair<!smt.array.array<!smt.int.int, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>>, !smt.utils.pair<!smt.bv.bv<64>, !smt.bool>>) -> ()
