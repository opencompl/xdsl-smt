// RUN: xdsl-smt %s -p=dynamic-semantics,lower-to-smt | filecheck %s

func.func @f(%x: i32, %y: i32) -> (i32) {
  %z = arith.addi %x,%y: i32
  return %z: i32
}

  // Match the constant operations.
  pdl.pattern : benefit(1) {
    %i32_type = pdl.type: i32
    %lhs = pdl.operand
    %rhs = pdl.operand
    %root = pdl.operation "arith.addi" (%lhs,%rhs: !pdl.value,!pdl.value) -> (%i32_type: !pdl.type)
    pdl.rewrite %root {
      %smt_i32_type = pdl.type: !smt.bv.bv<32>
      %smt_bool_type = pdl.type: !smt.bool
      // Unpack the arguments
      %lhs_get_payload_op = pdl.operation "smt.utils.first" (%lhs: !pdl.value) -> (%smt_i32_type: !pdl.type)
      %lhs_get_poison_op = pdl.operation "smt.utils.second" (%lhs: !pdl.value) -> (%smt_bool_type: !pdl.type)
      %rhs_get_payload_op = pdl.operation "smt.utils.first" (%rhs: !pdl.value) -> (%smt_i32_type: !pdl.type)
      %rhs_get_poison_op = pdl.operation "smt.utils.second" (%rhs: !pdl.value) -> (%smt_bool_type: !pdl.type)
      %lhs_payload = pdl.result 0 of %lhs_get_payload_op
      %rhs_payload = pdl.result 0 of %rhs_get_payload_op
      %lhs_poison = pdl.result 0 of %lhs_get_poison_op
      %rhs_poison = pdl.result 0 of %rhs_get_poison_op
      // Compute the addition
      %get_payload_op = pdl.operation "smt.bv.add" (%lhs_payload,%rhs_payload: !pdl.value,!pdl.value) -> (%smt_i32_type: !pdl.type)
      %payload = pdl.result 0 of %get_payload_op
      // Compute the poison
      %get_poison_op = pdl.operation "smt.or" (%lhs_poison,%rhs_poison: !pdl.value,!pdl.value) -> (%smt_bool_type: !pdl.type)
      %poison = pdl.result 0 of %get_poison_op
      // Pack the result
      %smt_pair_type = pdl.type: !smt.utils.pair<!smt.bv.bv<32>,!smt.bool>
      %smt_pair_op = pdl.operation "smt.utils.pair"(%payload,%poison: !pdl.value,!pdl.value) -> (%smt_pair_type: !pdl.type)
      %smt_pair = pdl.result 0 of %smt_pair_op
      // Perform the replacement
      pdl.replace %root with %smt_pair_op
    }
  }

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, %y : !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, %1 : !effect.state):
// CHECK-NEXT:      %2 = "smt.utils.first"(%x) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %3 = "smt.utils.second"(%x) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %4 = "smt.utils.first"(%y) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %5 = "smt.utils.second"(%y) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>) -> !smt.bool
// CHECK-NEXT:      %6 = "smt.bv.add"(%2, %4) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
// CHECK-NEXT:      %7 = "smt.or"(%3, %5) : (!smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      %z = "smt.utils.pair"(%6, %7) : (!smt.bv.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
// CHECK-NEXT:      "smt.return"(%z, %1) : (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "f"} : () -> ((!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !effect.state) -> (!smt.utils.pair<!smt.bv.bv<32>, !smt.bool>, !effect.state))
// CHECK-NEXT:    pdl.pattern : benefit(1) {
// CHECK-NEXT:      %i32_type = pdl.type {__operand_types = []} : i32
// CHECK-NEXT:      %lhs = pdl.operand {__operand_types = []}
// CHECK-NEXT:      %rhs = pdl.operand {__operand_types = []}
// CHECK-NEXT:      %root = pdl.operation "arith.addi" (%lhs, %rhs : !pdl.value, !pdl.value) -> (%i32_type : !pdl.type)
// CHECK-NEXT:      pdl.rewrite %root {
// CHECK-NEXT:        %smt_i32_type = pdl.type {__operand_types = []} : !smt.bv.bv<32>
// CHECK-NEXT:        %smt_bool_type = pdl.type {__operand_types = []} : !smt.bool
// CHECK-NEXT:        %lhs_get_payload_op = pdl.operation "smt.utils.first" (%lhs : !pdl.value) -> (%smt_i32_type : !pdl.type)
// CHECK-NEXT:        %lhs_get_poison_op = pdl.operation "smt.utils.second" (%lhs : !pdl.value) -> (%smt_bool_type : !pdl.type)
// CHECK-NEXT:        %rhs_get_payload_op = pdl.operation "smt.utils.first" (%rhs : !pdl.value) -> (%smt_i32_type : !pdl.type)
// CHECK-NEXT:        %rhs_get_poison_op = pdl.operation "smt.utils.second" (%rhs : !pdl.value) -> (%smt_bool_type : !pdl.type)
// CHECK-NEXT:        %lhs_payload = pdl.result 0 of %lhs_get_payload_op {__operand_types = [!pdl.operation]}
// CHECK-NEXT:        %rhs_payload = pdl.result 0 of %rhs_get_payload_op {__operand_types = [!pdl.operation]}
// CHECK-NEXT:        %lhs_poison = pdl.result 0 of %lhs_get_poison_op {__operand_types = [!pdl.operation]}
// CHECK-NEXT:        %rhs_poison = pdl.result 0 of %rhs_get_poison_op {__operand_types = [!pdl.operation]}
// CHECK-NEXT:        %get_payload_op = pdl.operation "smt.bv.add" (%lhs_payload, %rhs_payload : !pdl.value, !pdl.value) -> (%smt_i32_type : !pdl.type)
// CHECK-NEXT:        %payload = pdl.result 0 of %get_payload_op {__operand_types = [!pdl.operation]}
// CHECK-NEXT:        %get_poison_op = pdl.operation "smt.or" (%lhs_poison, %rhs_poison : !pdl.value, !pdl.value) -> (%smt_bool_type : !pdl.type)
// CHECK-NEXT:        %poison = pdl.result 0 of %get_poison_op {__operand_types = [!pdl.operation]}
// CHECK-NEXT:        %smt_pair_type = pdl.type {__operand_types = []} : !smt.utils.pair<!smt.bv.bv<32>, !smt.bool>
// CHECK-NEXT:        %smt_pair_op = pdl.operation "smt.utils.pair" (%payload, %poison : !pdl.value, !pdl.value) -> (%smt_pair_type : !pdl.type)
// CHECK-NEXT:        %smt_pair = pdl.result 0 of %smt_pair_op {__operand_types = [!pdl.operation]}
// CHECK-NEXT:        pdl.replace %root with  %smt_pair_op {__operand_types = [!pdl.operation, !pdl.operation]}
// CHECK-NEXT:      } attributes {__operand_types = [!pdl.operation]}
// CHECK-NEXT:    }
// CHECK-NEXT:  }
