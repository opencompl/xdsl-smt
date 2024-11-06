// RUN: xdsl-smt %s -p=lower-memory-to-array | filecheck %s
// RUN: xdsl-smt %s -p=lower-memory-to-array,lower-pairs -t=smt | z3 -in

%memory = "smt.declare_const"() : () -> !memory.memory
%block_id = "smt.declare_const"() : () -> !memory.block_id
%new_block = "smt.declare_const"() : () -> !memory.block

%block = memory.get_block %memory[%block_id]
// CHECK:    %block = smt.array.select %memory[%block_id] : !smt.array.array<!smt.int.int, !smt.utils.pair<!smt.array.array<!smt.bv.bv<64>, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>>, !smt.utils.pair<!smt.bv.bv<64>, !smt.bool>>>
%new_memory = memory.set_block %new_block, %memory[%block_id]
// CHECK-NEXT:    %new_memory = smt.array.store %memory[%block_id], %new_block : !smt.array.array<!smt.int.int, !smt.utils.pair<!smt.array.array<!smt.bv.bv<64>, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>>, !smt.utils.pair<!smt.bv.bv<64>, !smt.bool>>>

"test.op"(%block, %new_memory) : (!memory.block, !memory.memory) -> ()
