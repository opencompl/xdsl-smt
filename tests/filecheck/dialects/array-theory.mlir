// RUN: xdsl-smt "%s" | xdsl-smt | filecheck "%s"

builtin.module {
    %array = "smt.declare_const"() : () -> !smt.array.array<!smt.int.int, !smt.bool>
    %index = "smt.declare_const"() : () -> !smt.int.int
    %value = "smt.declare_const"() : () -> !smt.bool
    // CHECK:         %array = "smt.declare_const"() : () -> !smt.array.array<!smt.int.int, !smt.bool>
    // CHECK-NEXT:    %index = "smt.declare_const"() : () -> !smt.int.int
    // CHECK-NEXT:    %value = "smt.declare_const"() : () -> !smt.bool

    %read_value = smt.array.select %array[%index] : !smt.array.array<!smt.int.int, !smt.bool>
    %write_value = smt.array.store %array[%index], %value : !smt.array.array<!smt.int.int, !smt.bool>
    // CHECK-NEXT:    %read_value = smt.array.select %array[%index] : !smt.array.array<!smt.int.int, !smt.bool>
    // CHECK-NEXT:    %write_value = smt.array.store %array[%index], %value : !smt.array.array<!smt.int.int, !smt.bool>
}
