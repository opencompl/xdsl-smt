// RUN: xdsl-smt "%s" -t mlir -p=pdl-to-smt,lower-effects,canonicalize-smt -t smt | z3 -in | filecheck %s

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

// CHECK: unsat
