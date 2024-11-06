// RUN: xdsl-smt %s -p=lower-memory-to-array,lower-pairs | filecheck %s
// RUN: xdsl-smt %s -p=lower-memory-to-array,lower-pairs -t=smt | z3 -in

%bytes = "smt.declare_const"() : () -> !memory.bytes
%index = "smt.declare_const"() : () -> !smt.bv.bv<64>
%value = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>

%old_value = memory.read_bytes %bytes[%index] : !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>
// CHECK:    %old_value = smt.array.select %bytes[%index] : !smt.array.array<!smt.bv.bv<64>, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>>
%new_bytes = memory.write_bytes %value, %bytes[%index] : !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>
// CHECK-NEXT:    %new_bytes = smt.array.store %bytes[%index], %value : !smt.array.array<!smt.bv.bv<64>, !smt.utils.pair<!smt.bv.bv<8>, !smt.bool>>

"test.op"(%old_value, %new_bytes) : (!smt.utils.pair<!smt.bv.bv<8>, !smt.bool>, !memory.bytes) -> ()
