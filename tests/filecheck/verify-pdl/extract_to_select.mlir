// RUN: verify-pdl "%s" | filecheck "%s"

builtin.module {
  pdl.pattern @ExtractToSelect : benefit(0) {
    %type = pdl.type : i8
    %extract_type = pdl.type : i4

    %operand = pdl.operand : %type
    %low_bit_attr = pdl.attribute = 2 : i32

    %extract = pdl.operation "comb.extract"(%operand : !pdl.value) { "low_bit" = %low_bit_attr} -> (%extract_type : !pdl.type)

    pdl.rewrite %extract {
      %shift_cst_attr = pdl.attribute = 2 : i8 
      %shift_cst_op = pdl.operation "hw.constant" { "value" = %shift_cst_attr } -> (%type : !pdl.type)
      %shift_cst_res = pdl.result 0 of %shift_cst_op
      %shift_op = pdl.operation "comb.shru"(%operand, %shift_cst_res : !pdl.value, !pdl.value) -> (%type : !pdl.type)
      %shift_res = pdl.result 0 of %shift_op

      %new_extract_cst = pdl.attribute = 0 : i32
      %new_extract_op = pdl.operation "comb.extract"(%shift_res : !pdl.value) { "low_bit" = %new_extract_cst } -> (%extract_type : !pdl.type)

      pdl.replace %extract with %new_extract_op
    }
  }
}

// CHECK: All patterns are sound
