// RUN: xdsl-smt %s -p=lower-memory-to-array | filecheck %s
// RUN: xdsl-smt %s -p=lower-memory-to-array,lower-pairs -t=smt | z3 -in

%memory = "smt.declare_const"() : () -> !memory.memory
// CHECK:         %memory = "smt.declare_const"() : () -> !smt.array.array<!smt.int.int, !smt.utils.pair<!smt.array.array<!smt.int.int, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>>, !smt.utils.pair<!smt.bv.bv<64>, !smt.bool>>>
%block_id = "smt.declare_const"() : () -> !memory.block_id
// CHECK-NEXT:    %block_id = "smt.declare_const"() : () -> !smt.int.int
%block = "smt.declare_const"() : () -> !memory.block
// CHECK-NEXT:    %block = "smt.declare_const"() : () -> !smt.utils.pair<!smt.array.array<!smt.int.int, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>>, !smt.utils.pair<!smt.bv.bv<64>, !smt.bool>>
%bytes = "smt.declare_const"() : () -> !memory.bytes
// CHECK-NEXT:    %bytes = "smt.declare_const"() : () -> !smt.array.array<!smt.int.int, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>>
