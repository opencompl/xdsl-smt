// RUN: xdsl-smt %s -p=lower-memory-to-array,lower-pairs | filecheck %s
// RUN: xdsl-smt %s -p=lower-memory-to-array,lower-pairs -t=smt | z3 -in

%memory = "smt.declare_const"() : () -> !memory.memory

%block_id1, %memory2 = memory.get_fresh_block_id %memory
%block_id2, %memory3 = memory.get_fresh_block_id %memory2
%block_id3, %memory4 = memory.get_fresh_block_id %memory3

// CHECK:         %block_id1 = "smt.int.constant"() {value = 0 : ui128} : () -> !smt.int.int
// CHECK-NEXT:    %block_id2 = "smt.int.constant"() {value = 1 : ui128} : () -> !smt.int.int
// CHECK-NEXT:    %block_id3 = "smt.int.constant"() {value = 2 : ui128} : () -> !smt.int.int

"test.op"(%block_id1, %block_id2, %block_id3, %memory4) : (!memory.block_id, !memory.block_id, !memory.block_id, !memory.memory) -> ()
