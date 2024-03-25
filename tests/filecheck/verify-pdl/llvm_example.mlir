// RUN: verify-pdl "%s" | filecheck "%s"

// XOR(AND(x, C), (C + 1)) → NEG(OR(x, ∼C)) when C is even

builtin.module {
  pdl.pattern @llvm_example : benefit(0) {
    %type = pdl.type : !transfer.integer

    // Get the constants 0, 1, C and C + 1 as attributes
    %C_attr = pdl.attribute : %type
    %zero_attr = pdl.apply_native_rewrite "get_zero_attr"(%type : !pdl.type) : !pdl.attribute
    %one_attr = pdl.apply_native_rewrite "get_one_attr"(%type : !pdl.type) : !pdl.attribute
    %C_plus_one_attr = pdl.apply_native_rewrite "addi"(%C_attr, %one_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute

    // Get C and C + 1 as SSA values
    %C_op = pdl.operation "arith.constant" {"value" = %C_attr} -> (%type: !pdl.type)
    %C = pdl.result 0 of %C_op
    %C_plus_1_op = pdl.operation "arith.constant" {"value" = %C_plus_one_attr} -> (%type: !pdl.type)
    %C_plus_1 = pdl.result 0 of %C_plus_1_op

    // Check that C is even by checking that C & 1 == 0
    %C_and_one_attr = pdl.apply_native_rewrite "andi"(%C_attr, %one_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute
    pdl.apply_native_constraint "is_attr_equal"(%C_and_one_attr, %zero_attr : !pdl.attribute, !pdl.attribute)

    // Get the matching DAG
    %x = pdl.operand : %type
    %and = pdl.operation "arith.andi"(%x, %C : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    %and_val = pdl.result 0 of %and
    %xor = pdl.operation "arith.xori"(%and_val, %C_plus_1 : !pdl.value, !pdl.value) -> (%type : !pdl.type)

    pdl.rewrite %xor {
      // Compute `~C` by doing `C ^ -1`
      %-one_attr = pdl.apply_native_rewrite "get_minus_one_attr"(%type : !pdl.type) : !pdl.attribute
      %not_C_attr = pdl.apply_native_rewrite "xori"(%C_attr, %-one_attr : !pdl.attribute, !pdl.attribute) : !pdl.attribute

      // Get `~C` as an SSA value
      %not_C_op = pdl.operation "arith.constant" {"value" = %not_C_attr} -> (%type: !pdl.type)
      %not_C = pdl.result 0 of %not_C_op

      // Get the rewritten DAG
      %or = pdl.operation "arith.ori"(%x, %not_C : !pdl.value, !pdl.value) -> (%type : !pdl.type)
      %or_val = pdl.result 0 of %or
      %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%type: !pdl.type)
      %zero = pdl.result 0 of %zero_op
      %neg = pdl.operation "arith.subi"(%zero, %or_val : !pdl.value, !pdl.value) -> (%type : !pdl.type)

      pdl.replace %xor with %neg
    }
  }
}

// CHECK: All patterns are sound
