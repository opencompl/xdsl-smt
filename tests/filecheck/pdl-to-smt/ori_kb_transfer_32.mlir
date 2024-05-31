// RUN: xdsl-smt "%s" -t mlir -p=pdl-to-smt,canonicalize-smt | filecheck "%s"


builtin.module {
  pdl.pattern @addi_analysis : benefit(0) {
    // Get an i32 type
    %type = pdl.type : i32

    // Get the two operation operands
    %lhs = pdl.operand : %type
    %rhs = pdl.operand : %type

    // Get the operands analysis values
    %lhs_zeros, %lhs_ones = "pdl.dataflow.get"(%lhs) {"domain_name" = "kb"} : (!pdl.value) -> (!transfer.integer, !transfer.integer)
    %rhs_zeros, %rhs_ones = "pdl.dataflow.get"(%rhs) {"domain_name" = "kb"} : (!pdl.value) -> (!transfer.integer, !transfer.integer)

    // Get an add operation that takes both operands
    %op = pdl.operation "arith.ori"(%lhs, %rhs : !pdl.value, !pdl.value) -> (%type : !pdl.type)

    pdl.rewrite %op {
        // Compute the known bits of the result
        %res_zeros = "transfer.and"(%lhs_zeros, %rhs_zeros) : (!transfer.integer, !transfer.integer) -> !transfer.integer
        %res_ones = "transfer.or"(%lhs_ones, %rhs_ones) : (!transfer.integer, !transfer.integer) -> !transfer.integer

        // Get the value of the operation result
        %res = pdl.result 0 of %op

        // Attach it to the operation
        "pdl.dataflow.attach"(%res, %res_zeros, %res_ones) {"domain_name" = "kb"} : (!pdl.value, !transfer.integer, !transfer.integer) -> ()
    }
  }
}


// CHECK:      builtin.module {
// CHECK-NEXT:   %lhs = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
// CHECK-NEXT:   %rhs = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
// CHECK-NEXT:   %lhs_zeros = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %lhs_ones = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %0 = "smt.utils.first"(%lhs) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %1 = "smt.bv.and"(%0, %lhs_zeros) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %2 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %3 = "smt.eq"(%1, %2) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:   %4 = "smt.bv.and"(%0, %lhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %5 = "smt.eq"(%4, %lhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:   %6 = "smt.and"(%3, %5) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   %rhs_zeros = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %rhs_ones = "smt.declare_const"() : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %7 = "smt.utils.first"(%rhs) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %8 = "smt.bv.and"(%7, %rhs_zeros) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %9 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %10 = "smt.eq"(%8, %9) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:   %11 = "smt.bv.and"(%7, %rhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %12 = "smt.eq"(%11, %rhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:   %13 = "smt.and"(%10, %12) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   %14 = "smt.utils.first"(%lhs) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %15 = "smt.utils.first"(%rhs) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %16 = "smt.bv.or"(%14, %15) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %res_zeros = "smt.bv.and"(%lhs_zeros, %rhs_zeros) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %res_ones = "smt.bv.or"(%lhs_ones, %rhs_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %17 = "smt.bv.and"(%16, %res_zeros) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %18 = "smt.bv.constant"() {"value" = #smt.bv.bv_val<0: 32>} : () -> !smt.bv.bv<32>
// CHECK-NEXT:   %19 = "smt.eq"(%17, %18) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:   %20 = "smt.bv.and"(%16, %res_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:   %21 = "smt.eq"(%20, %res_ones) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
// CHECK-NEXT:   %22 = "smt.and"(%19, %21) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   %23 = "smt.not"(%22) : (!smt.bool) -> !smt.bool
// CHECK-NEXT:   %24 = "smt.and"(%6, %13) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   %25 = "smt.and"(%24, %23) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:   "smt.assert"(%25) : (!smt.bool) -> ()
// CHECK-NEXT:   "smt.check_sat"() : () -> ()
// CHECK-NEXT: }
