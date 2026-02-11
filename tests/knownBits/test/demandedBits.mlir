"builtin.module"() ({
"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %result = "transfer.cmp"(%arg0_0, %arg0_0){predicate=0:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>) -> i1, sym_name = "getConstraint"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %result = "transfer.cmp"(%arg0_0, %arg0_0){predicate=0:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    "func.return"(%result) : (i1) -> ()
}) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>) -> i1, sym_name = "getInstanceConstraint"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %andi = "transfer.and"(%arg0_0, %arg0_1) : (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.cmp"(%andi, %const0){predicate=0:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1, sym_name = "isValidKnownBit"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %inst: !transfer.integer<8>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %neg_inst = "transfer.neg"(%inst) : (!transfer.integer<8>) -> !transfer.integer<8>
    %or1 = "transfer.or"(%neg_inst,%arg0_0): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %or2 = "transfer.or"(%inst,%arg0_1): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %cmp1="transfer.cmp"(%or1,%neg_inst){predicate=0:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    %cmp2="transfer.cmp"(%or2,%inst){predicate=0:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    %result="arith.andi"(%cmp1,%cmp2):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1, sym_name = "inKnownBits"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %eqinst="transfer.and"(%arg0_0,%inst):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %eqinst1="transfer.and"(%arg0_0,%inst1):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %eq = "transfer.cmp"(%eqinst, %eqinst1){predicate=0:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    "func.return"(%eq) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1, sym_name = "inSameEq"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>):
    %concrete_res0 = "transfer.xor"(%inst,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.xor"(%inst1,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %absres =  "func.call"(%arg0) {callee = @XORImpl0} : (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>

    %precond = "func.call"(%absres, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %postcond = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>,!transfer.integer<8>) -> i1, sym_name = "counterXor"} : () -> ()

"func.func"() ({
  ^bb0(%absres:  !transfer.abs_value<[!transfer.integer<8>]>, %arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>):
    %concrete_res0 = "transfer.xor"(%inst,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.xor"(%inst1,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>

    %precond = "func.call"(%absres, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %postcond = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>,!transfer.integer<8>) -> i1,other_operand=[4], sym_name = "precision_counterXor", abs_input=1} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>):
    "func.return"(%arg0) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>, sym_name = "XORImpl1", operationNo=0, applied_to=["comb.xor"], is_forward=false, CPPCLASS=["circt::comb::XorOp"], precision_util="precision_counterXor", soundness_counterexample="counterXor"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>):
    "func.return"(%arg0) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>, sym_name = "XORImpl0", operationNo=1, applied_to=["comb.xor"], is_forward=false, CPPCLASS=["circt::comb::XorOp"], precision_util="precision_counterXor", soundness_counterexample="counterXor"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.and"(%inst,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.and"(%inst1,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @AndImpl0} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1, sym_name = "counterAnd0"} : () -> ()


"func.func"() ({
  ^bb0(%abs_arg: !transfer.abs_value<[!transfer.integer<8>]>, %arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.and"(%inst,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.and"(%inst1,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1

    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1,other_operand=[4,6], sym_name = "precision_counterAnd0", abs_input=1} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %op0_0 = "transfer.get"(%op0) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op0_1 = "transfer.get"(%op0) {index=1:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %neg_op1_0 = "transfer.neg"(%op1_0) : (!transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.and"(%neg_op1_0, %arg0_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_1) : (!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>, operationNo=0, sym_name = "AndImpl0", applied_to=["comb.and"], CPPCLASS=["circt::comb::AndOp"],is_forward=false, precision_util="precision_counterAnd0", soundness_counterexample="counterAnd0"} : () -> ()


"func.func"() ({
  ^bb0(%abs_arg: !transfer.abs_value<[!transfer.integer<8>]>, %arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.and"(%operand,%inst):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.and"(%operand,%inst1):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1,other_operand=[4,6], sym_name = "precision_counterAnd1", abs_input=1} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.and"(%operand,%inst):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.and"(%operand,%inst1):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @AndImpl1} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1, sym_name = "counterAnd1"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %op0_0 = "transfer.get"(%op0) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op0_1 = "transfer.get"(%op0) {index=1:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %neg_op1_0 = "transfer.neg"(%op1_0) : (!transfer.integer<8>) -> !transfer.integer<8>
    %and_neg = "transfer.and"(%neg_op1_0, %op0_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %neg_and= "transfer.neg"(%and_neg) : (!transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.and"(%neg_and, %arg0_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_1) : (!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>, sym_name = "AndImpl1", operationNo=1, applied_to=["comb.and"], CPPCLASS=["circt::comb::AndOp"],is_forward=false, precision_util="precision_counterAnd1", soundness_counterexample="counterAnd1"} : () -> ()


"func.func"() ({
  ^bb0(%abs_arg: !transfer.abs_value<[!transfer.integer<8>]>, %arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.or"(%inst,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.or"(%inst1,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1,other_operand=[4,6], abs_input=1, sym_name = "precision_counterOr0"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.or"(%inst,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.or"(%inst1,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @OrImpl0} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1, sym_name = "counterOr0"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %op0_0 = "transfer.get"(%op0) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op0_1 = "transfer.get"(%op0) {index=1:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %neg_op1_1 = "transfer.neg"(%op1_1) : (!transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.and"(%neg_op1_1, %arg0_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_1) : (!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>, sym_name = "OrImpl0", applied_to=["comb.or"], operationNo=0, CPPCLASS=["circt::comb::OrOp"],is_forward=false, precision_util="precision_counterOr0", soundness_counterexample="counterOr0"} : () -> ()


"func.func"() ({
  ^bb0(%abs_arg: !transfer.abs_value<[!transfer.integer<8>]>, %arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.or"(%operand, %inst):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.or"(%operand, %inst1):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1,other_operand=[4,6],abs_input=1, sym_name = "precision_counterOr1"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.or"(%operand, %inst):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.or"(%operand, %inst1):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @OrImpl1} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1, sym_name = "counterOr1"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %op0_0 = "transfer.get"(%op0) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op0_1 = "transfer.get"(%op0) {index=1:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %neg_op1_1 = "transfer.neg"(%op1_1) : (!transfer.integer<8>) -> !transfer.integer<8>
    %and_neg = "transfer.and"(%neg_op1_1, %op0_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %neg_and = "transfer.neg"(%and_neg) : (!transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.and"(%neg_and, %arg0_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_1) : (!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>, sym_name = "OrImpl1", applied_to=["comb.or"], operationNo=1, CPPCLASS=["circt::comb::OrOp"],is_forward=false, precision_util="precision_counterOr1", soundness_counterexample="counterOr1"} : () -> ()

"func.func"() ({
  ^bb0(%operationNo:i1, %arg0: !transfer.abs_value<[!transfer.integer<8>]>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %carryZero: !transfer.integer<8>, %carryOne: !transfer.integer<8>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %op0_0 = "transfer.get"(%op0) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op0_1 = "transfer.get"(%op0) {index=1:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>

    %and_0_0 = "transfer.and"(%op0_0, %op1_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %and_1_1 = "transfer.and"(%op0_1, %op1_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %bound = "transfer.or"(%and_0_0, %and_1_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>

    %rbound = "transfer.reverse_bits"(%bound) : (!transfer.integer<8>) -> !transfer.integer<8>
    %rarg0_0 = "transfer.reverse_bits"(%arg0_0) : (!transfer.integer<8>) -> !transfer.integer<8>
    %neg_rbound = "transfer.neg"(%rbound) : (!transfer.integer<8>) -> !transfer.integer<8>
    %or_rarg0_0_neg = "transfer.or"(%rarg0_0, %neg_rbound) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %rprop = "transfer.add"(%rarg0_0, %or_rarg0_0_neg) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %racarry = "transfer.xor"(%rprop, %neg_rbound) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %acarry = "transfer.reverse_bits"(%racarry) : (!transfer.integer<8>) -> !transfer.integer<8>

    %neg_op0_0 = "transfer.neg"(%op0_0) : (!transfer.integer<8>) -> !transfer.integer<8>
    %neg_op0_1 = "transfer.neg"(%op0_1) : (!transfer.integer<8>) -> !transfer.integer<8>
    %neg_op1_0 = "transfer.neg"(%op1_0) : (!transfer.integer<8>) -> !transfer.integer<8>
    %neg_op1_1 = "transfer.neg"(%op1_1) : (!transfer.integer<8>) -> !transfer.integer<8>

    %or_0_0_neg_1 = "transfer.or"(%op0_0, %neg_op1_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %or_0_neg_0_1 = "transfer.or"(%neg_op0_0, %op1_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %or_1_0_neg_1 = "transfer.or"(%op0_1, %neg_op1_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %or_1_neg_0_1 = "transfer.or"(%neg_op0_1, %op1_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %neededToMaintainCarryZero ="transfer.select"(%operationNo, %or_0_neg_0_1, %or_0_0_neg_1) : (i1, !transfer.integer<8>, !transfer.integer<8>) ->!transfer.integer<8>
    %neededToMaintainCarryOne ="transfer.select"(%operationNo, %or_1_neg_0_1, %or_1_0_neg_1) : (i1, !transfer.integer<8>, !transfer.integer<8>) ->!transfer.integer<8>

    %one="transfer.constant"(%arg0_0){value=1:index}:(!transfer.integer<8>)->!transfer.integer<8>
    %negCarryZero="transfer.sub"(%one,%carryZero):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %possibleSumZeroTmp = "transfer.add" (%neg_op0_0,%neg_op1_0):(!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %possibleSumZero="transfer.add"(%possibleSumZeroTmp,%negCarryZero): (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %neg_possibleSumZero = "transfer.neg"(%possibleSumZero) : (!transfer.integer<8>) -> !transfer.integer<8>
    %possibleSumOneTmp = "transfer.add" (%op0_1,%op1_1): (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %possibleSumOne="transfer.add"(%possibleSumOneTmp,%carryOne):(!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>


    %neededToMaintainCarry_0 = "transfer.or"(%neg_possibleSumZero, %neededToMaintainCarryZero) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %neededToMaintainCarry_1 = "transfer.or"(%possibleSumOne, %neededToMaintainCarryOne) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %neededToMaintainCarry = "transfer.and"(%neededToMaintainCarry_0, %neededToMaintainCarry_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>

    %carryAnd = "transfer.and"(%acarry, %neededToMaintainCarry) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.or"(%arg0_0, %carryAnd) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>

    %result = "transfer.make"(%result_1) : (!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (i1, !transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.integer<8>,!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>, sym_name = "determineLiveOperandBitsAddCarry"} : () -> ()

"func.func"() ({
  ^bb0(%abs_arg: !transfer.abs_value<[!transfer.integer<8>]>, %arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.add"(%inst, %operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.add"(%inst1, %operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1,other_operand=[4,6],abs_input=1, sym_name = "precision_counterAdd0"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.add"(%inst, %operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.add"(%inst1, %operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @AddImpl0} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1, sym_name = "counterAdd0"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1

    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %transfer_const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %transfer_const1 = "transfer.constant"(%arg0_0){value=1:index} : (!transfer.integer<8>) -> !transfer.integer<8>

    %result = "func.call"(%const0, %arg0, %op0, %op1, %transfer_const1, %transfer_const0) {callee = @determineLiveOperandBitsAddCarry} : (i1, !transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.integer<8>,!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>

    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>, sym_name = "AddImpl0", applied_to=["comb.add"], operationNo=0, CPPCLASS=["circt::comb::AddOp"],is_forward=false, precision_util="precision_counterAdd0", soundness_counterexample="counterAdd0"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.add"(%operand, %inst):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.add"(%operand, %inst1):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @AddImpl1} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1, sym_name = "counterAdd1"} : () -> ()

"func.func"() ({
  ^bb0(%abs_arg: !transfer.abs_value<[!transfer.integer<8>]>, %arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.add"(%operand, %inst):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.add"(%operand, %inst1):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1,other_operand=[4,6],abs_input=1, sym_name = "precision_counterAdd1"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1

    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %transfer_const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %transfer_const1 = "transfer.constant"(%arg0_0){value=1:index} : (!transfer.integer<8>) -> !transfer.integer<8>

    %result = "func.call"(%const1, %arg0, %op0, %op1, %transfer_const1, %transfer_const0) {callee = @determineLiveOperandBitsAddCarry} : (i1, !transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.integer<8>,!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>

    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>, sym_name = "AddImpl1", applied_to=["comb.add"], operationNo=0, CPPCLASS=["circt::comb::AddOp"],is_forward=false, precision_util="precision_counterAdd1", soundness_counterexample="counterAdd1"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.sub"(%inst, %operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.sub"(%inst1, %operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @SubImpl0} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1, sym_name = "counterSub0"} : () -> ()

"func.func"() ({
  ^bb0(%abs_arg: !transfer.abs_value<[!transfer.integer<8>]>, %arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.sub"(%inst, %operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.sub"(%inst1, %operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1,other_operand=[4,6], abs_input=1, sym_name = "precision_counterSub0"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1

    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %newOp="transfer.make"(%op1_1,%op1_0):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>
    %transfer_const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %transfer_const1 = "transfer.constant"(%arg0_0){value=1:index} : (!transfer.integer<8>) -> !transfer.integer<8>

    %result = "func.call"(%const0, %arg0, %op0, %newOp, %transfer_const0, %transfer_const1) {callee = @determineLiveOperandBitsAddCarry} : (i1, !transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.integer<8>,!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>

    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>, sym_name = "SubImpl0", applied_to=["comb.sub"], operationNo=0, CPPCLASS=["circt::comb::SubOp"],is_forward=false, precision_util="precision_counterSub0", soundness_counterexample="counterSub0"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.sub"(%operand, %inst):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.sub"(%operand, %inst1):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @SubImpl1} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1, sym_name = "counterSub1"} : () -> ()

"func.func"() ({
  ^bb0(%abs_arg: !transfer.abs_value<[!transfer.integer<8>]>, %arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %concrete_res0 = "transfer.sub"(%operand, %inst):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.sub"(%operand, %inst1):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1,other_operand=[4,6], abs_input=1, sym_name = "precision_counterSub1"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %op1: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1

    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %newOp="transfer.make"(%op1_1,%op1_0):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>
    %transfer_const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %transfer_const1 = "transfer.constant"(%arg0_0){value=1:index} : (!transfer.integer<8>) -> !transfer.integer<8>

    %result = "func.call"(%const1, %arg0, %op0, %newOp, %transfer_const0, %transfer_const1) {callee = @determineLiveOperandBitsAddCarry} : (i1, !transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.integer<8>,!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>

    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>]>, sym_name = "SubImpl1", applied_to=["comb.sub"], CPPCLASS=["circt::comb::SubOp"], operationNo=1,is_forward=false, precision_util="precision_counterSub1", soundness_counterexample="counterSub1"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.tuple<[i1,i1]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.tuple<[i1,i1]>) -> i1
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.tuple<[i1,i1]>) -> i1
    %arg0_0_arith = "transfer.add_poison"(%arg0_0): (i1) -> i1
    %arg0_1_arith = "transfer.add_poison"(%arg0_1): (i1) -> i1
    %cmp_res = "arith.xori"(%arg0_0_arith,%arg0_1_arith) {"predicate" = 0 : i64} : (i1,i1) -> i1
    "func.return"(%cmp_res) : (i1) -> ()
  }) {function_type = (!transfer.tuple<[i1,i1]>) -> i1, sym_name = "isConstant_i1"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[i1,i1]>):
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[i1,i1]>) -> i1
    %arg0_1_arith = "transfer.add_poison"(%arg0_1): (i1) -> i1
    "func.return"(%arg0_1_arith) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[i1,i1]>) -> i1, sym_name = "getConstant_i1"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.tuple<[i1,i1]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.tuple<[i1,i1]>) -> i1
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.tuple<[i1,i1]>) -> i1
    %arg0_0_arith = "transfer.add_poison"(%arg0_0): (i1) -> i1
    %arg0_1_arith = "transfer.add_poison"(%arg0_1): (i1) -> i1
    %andi = "arith.andi"(%arg0_0_arith, %arg0_1_arith) : (i1,i1) -> i1
    %const0_i1 = "arith.constant"() {value=0:i1}: () -> i1
    %result = "arith.cmpi"(%andi, %const0_i1){predicate=0:i64}: (i1,i1) -> i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.tuple<[i1,i1]>) -> i1, sym_name = "isValidKnownBiti1"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.tuple<[i1,i1]>, %inst: i1):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.tuple<[i1,i1]>) -> i1
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.tuple<[i1,i1]>) -> i1
    %arg0_0_arith = "transfer.add_poison"(%arg0_0): (i1) -> i1
    %arg0_1_arith = "transfer.add_poison"(%arg0_1): (i1) -> i1
    %const1_i1 = "arith.constant"() {value=1:i1}: () -> i1
    %neg_inst = "arith.xori"(%inst, %const1_i1) : (i1,i1) -> i1
    %or1 = "arith.ori"(%neg_inst,%arg0_0_arith): (i1,i1) -> i1
    %or2 = "arith.ori"(%inst,%arg0_1_arith): (i1,i1) -> i1
    %cmp1="arith.cmpi"(%or1,%neg_inst){predicate=0:i64}: (i1,i1) -> i1
    %cmp2="arith.cmpi"(%or2,%inst){predicate=0:i64}: (i1,i1) -> i1
    %result="arith.andi"(%cmp1,%cmp2):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.tuple<[i1,i1]>, i1) -> i1, sym_name = "inKnownBitsi1"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %cond: !transfer.tuple<[i1,i1]>, %branchNo: i1 ):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer<8>) -> !transfer.integer<8>

    %cond_const = "func.call"(%cond) {callee = @isConstant_i1} : (!transfer.tuple<[i1,i1]>) -> i1
    %const0_i1 = "arith.constant"() {value=0:i1}: () -> i1
    %not_cond_const = "arith.cmpi"(%cond_const, %const0_i1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %cond_val = "func.call"(%cond) {callee = @getConstant_i1} : (!transfer.tuple<[i1,i1]>) -> i1

    %cond_eq_branch = "arith.xori"(%cond_val, %branchNo) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %cond_eq_branch_or_not_const = "arith.ori"(%cond_eq_branch, %not_cond_const) : (i1, i1) -> i1
    %cond_res = "transfer.select"(%cond_eq_branch_or_not_const, %arg0_0, %const0): (i1, !transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>

    %result = "transfer.make"(%cond_res) : (!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[i1,i1]>, i1) -> !transfer.abs_value<[!transfer.integer<8>]>, sym_name = "MUXImplHelper"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>,%cond_kb: !transfer.tuple<[i1,i1]>, %cond:i1, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>):
    %concrete_res0 = "transfer.select"(%cond, %inst,%operand):(i1,!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.select"(%cond, %inst1,%operand):(i1,!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %absres =  "func.call"(%arg0, %cond_kb) {callee = @MUXImpl0} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[i1,i1]>) -> !transfer.abs_value<[!transfer.integer<8>]>

    %precond = "func.call"(%absres, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %postcond = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1

    %isKnownBits = "func.call"(%cond_kb) {callee = @isValidKnownBiti1} : (!transfer.tuple<[i1,i1]>) -> i1
    %inKnownBits = "func.call"(%cond_kb,%cond) {callee = @inKnownBitsi1} : (!transfer.tuple<[i1,i1]>,i1) -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result0="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result1="arith.andi"(%isKnownBits,%result0):(i1,i1)->i1
    %result="arith.andi"(%inKnownBits,%result1):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.tuple<[i1,i1]>,i1 , !transfer.integer<8>, !transfer.integer<8>,!transfer.integer<8>) -> i1, sym_name = "counterMux0"} : () -> ()

"func.func"() ({
  ^bb0(%absres: !transfer.abs_value<[!transfer.integer<8>]>, %arg0: !transfer.abs_value<[!transfer.integer<8>]>,%cond_kb: !transfer.tuple<[i1,i1]>, %cond:i1, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>):
    %concrete_res0 = "transfer.select"(%cond, %inst,%operand):(i1,!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.select"(%cond, %inst1,%operand):(i1,!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>

    %precond = "func.call"(%absres, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %postcond = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1

    %isKnownBits = "func.call"(%cond_kb) {callee = @isValidKnownBiti1} : (!transfer.tuple<[i1,i1]>) -> i1
    %inKnownBits = "func.call"(%cond_kb,%cond) {callee = @inKnownBitsi1} : (!transfer.tuple<[i1,i1]>,i1) -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result0="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result1="arith.andi"(%isKnownBits,%result0):(i1,i1)->i1
    %result="arith.andi"(%inKnownBits,%result1):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>]>, !transfer.tuple<[i1,i1]>,i1 , !transfer.integer<8>, !transfer.integer<8>,!transfer.integer<8>) -> i1,other_operand=[6], abs_input=1, sym_name = "precision_counterMux0"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>,%cond: !transfer.tuple<[i1,i1]> ):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1

    %result = "func.call"(%arg0, %cond, %const0) {callee = @MUXImplHelper} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[i1,i1]>, i1) -> !transfer.abs_value<[!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[i1,i1]>) -> !transfer.abs_value<[!transfer.integer<8>]>
, precision_util="precision_counterMux0", soundness_counterexample="counterMux0", sym_name = "MUXImpl0", operationNo=1,is_forward=false, applied_to=["comb.mux"], CPPCLASS=["circt::comb::MuxOp"]} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>,%cond_kb: !transfer.tuple<[i1,i1]>, %cond:i1, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>):
    %concrete_res0 = "transfer.select"(%cond, %operand,%inst):(i1,!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.select"(%cond, %operand,%inst1):(i1,!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %absres =  "func.call"(%arg0, %cond_kb) {callee = @MUXImpl1} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[i1,i1]>) -> !transfer.abs_value<[!transfer.integer<8>]>

    %precond = "func.call"(%absres, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %postcond = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1

    %isKnownBits = "func.call"(%cond_kb) {callee = @isValidKnownBiti1} : (!transfer.tuple<[i1,i1]>) -> i1
    %inKnownBits = "func.call"(%cond_kb,%cond) {callee = @inKnownBitsi1} : (!transfer.tuple<[i1,i1]>,i1) -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result0="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result1="arith.andi"(%isKnownBits,%result0):(i1,i1)->i1
    %result="arith.andi"(%inKnownBits,%result1):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.tuple<[i1,i1]>,i1 , !transfer.integer<8>, !transfer.integer<8>,!transfer.integer<8>) -> i1, sym_name = "counterMux1"} : () -> ()

"func.func"() ({
  ^bb0(%absres: !transfer.abs_value<[!transfer.integer<8>]>, %arg0: !transfer.abs_value<[!transfer.integer<8>]>,%cond_kb: !transfer.tuple<[i1,i1]>, %cond:i1, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>):
    %concrete_res0 = "transfer.select"(%cond, %operand,%inst):(i1,!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.select"(%cond, %operand,%inst1):(i1,!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>


    %precond = "func.call"(%absres, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %postcond = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1

    %isKnownBits = "func.call"(%cond_kb) {callee = @isValidKnownBiti1} : (!transfer.tuple<[i1,i1]>) -> i1
    %inKnownBits = "func.call"(%cond_kb,%cond) {callee = @inKnownBitsi1} : (!transfer.tuple<[i1,i1]>,i1) -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result0="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result1="arith.andi"(%isKnownBits,%result0):(i1,i1)->i1
    %result="arith.andi"(%inKnownBits,%result1):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>]>, !transfer.tuple<[i1,i1]>,i1 , !transfer.integer<8>, !transfer.integer<8>,!transfer.integer<8>) -> i1,other_operand=[6], abs_input=1, sym_name = "precision_counterMux1"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>,%cond: !transfer.tuple<[i1,i1]> ):
    %const1 = "arith.constant"() {value=1:i1}: () -> i1

    %result = "func.call"(%arg0, %cond, %const1) {callee = @MUXImplHelper} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[i1,i1]>, i1) -> !transfer.abs_value<[!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[i1,i1]>) -> !transfer.abs_value<[!transfer.integer<8>]>
  , precision_util="precision_counterMux1", soundness_counterexample="counterMux1",sym_name = "MUXImpl1", operationNo=2,is_forward=false, applied_to=["comb.mux"], CPPCLASS=["circt::comb::MuxOp"]} : () -> ()

"func.func"() ({
^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %op0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %len:!transfer.integer<8>,%low_bit :!transfer.integer<8>):
    %op0_0 = "transfer.get"(%op0) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %bitwidth = "transfer.get_bit_width"(%op0_0): (!transfer.integer<8>) -> !transfer.integer<8>
    %add_res = "transfer.add"(%len, %low_bit) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.cmp"(%add_res, %bitwidth){predicate=7:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1

    %low_bit_res = "transfer.cmp"(%low_bit, %bitwidth){predicate=6:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1

    %const1 = "transfer.constant"(%len){value=1:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %len_ge_1 = "transfer.cmp"(%len, %const1){predicate=9:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    %result_2="arith.andi"(%result_1,%len_ge_1):(i1,i1)->i1
    %result="arith.andi"(%result_2,%low_bit_res):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.integer<8>,!transfer.integer<8>) -> i1,
  sym_name = "EXTRACTAttrConstraint"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %len:!transfer.integer<8>, %low_bit :!transfer.integer<8>,  %inst1: !transfer.integer<8>):
    %concrete_res0 = "transfer.extract"(%inst,%len, %low_bit):(!transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %concrete_res1 = "transfer.extract"(%inst1,%len, %low_bit):(!transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %absres =  "func.call"(%arg0, %inst, %len, %low_bit) {callee = @EXTRACTImpl} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.integer<8>,!transfer.integer<8>,!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>

    %precond = "func.call"(%absres, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %postcond = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.integer<8>) -> i1, int_attr=[2,3], sym_name = "counterEXTRACT"} : () -> ()


"func.func"() ({
  ^bb0(%absres:!transfer.abs_value<[!transfer.integer<8>]>, %arg0: !transfer.abs_value<[!transfer.integer<8>]>, %len:!transfer.integer<8>, %low_bit :!transfer.integer<8>, %inst: !transfer.integer<8>,  %inst1: !transfer.integer<8>):
    %concrete_res0 = "transfer.extract"(%inst,%len, %low_bit):(!transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %concrete_res1 = "transfer.extract"(%inst1,%len, %low_bit):(!transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>

    %precond = "func.call"(%absres, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %postcond = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>,!transfer.integer<8>) -> i1, abs_input=1, int_attr=[2,3], sym_name = "precision_counterEXTRACT"} : () -> ()

"func.func"() ({
^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %op0: !transfer.integer<8>, %len:!transfer.integer<8>,%low_bit :!transfer.integer<8>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %const0 = "transfer.constant"(%op0){value=0:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %concat_res = "transfer.concat"(%const0, %arg0_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %shl_res = "transfer.shl"(%concat_res, %low_bit) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %bitwidth = "transfer.get_bit_width"(%op0): (!transfer.integer<8>) -> !transfer.integer<8>
    %result_0 = "transfer.extract"(%shl_res, %bitwidth, %const0) : (!transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>

    %result = "transfer.make"(%result_0) : (!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.integer<8>,!transfer.integer<8>,!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>,
  sym_name = "EXTRACTImpl", operationNo=0,is_forward=false,
  applied_to=["comb.extract"], CPPCLASS=["circt::comb::ExtractOp"], int_attr=[2,3], replace_int_attr=true, precision_util="precision_counterEXTRACT", soundness_counterexample="counterEXTRACT",
  int_attr_constraint="EXTRACTAttrConstraint"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>):
    %concrete_res0 = "transfer.concat"(%inst,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.concat"(%inst1,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %absres =  "func.call"(%arg0, %inst, %operand) {callee = @ConcatImpl0} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.integer<8>,!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>

    %precond = "func.call"(%absres, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %postcond = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>,!transfer.integer<8>) -> i1, sym_name = "counterConcat0"} : () -> ()

"func.func"() ({
  ^bb0(%absres: !transfer.abs_value<[!transfer.integer<8>]>, %arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>):
    %concrete_res0 = "transfer.concat"(%inst,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.concat"(%inst1,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>

    %precond = "func.call"(%absres, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %postcond = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>,!transfer.integer<8>) -> i1,other_operand=[4], abs_input=1,sym_name = "precision_counterConcat0"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %op0: !transfer.integer<8>, %op1: !transfer.integer<8>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %const0 = "transfer.constant"(%op0){value=0:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %bitwidth0 = "transfer.get_bit_width"(%op0): (!transfer.integer<8>) -> !transfer.integer<8>
    %bitwidth1 = "transfer.get_bit_width"(%op1): (!transfer.integer<8>) -> !transfer.integer<8>

    %result_0 = "transfer.extract"(%arg0_0, %bitwidth0, %bitwidth1) : (!transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_0) : (!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.integer<8>,
  !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>,
   sym_name = "ConcatImpl0", precision_util="precision_counterConcat0", soundness_counterexample="counterConcat0", applied_to=["comb.concat"], operationNo=0, CPPCLASS=["circt::comb::ConcatOp"],is_forward=false} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>):
    %concrete_res0 = "transfer.concat"(%inst,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.concat"(%inst1,%operand):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %absres =  "func.call"(%arg0, %inst, %operand) {callee = @ConcatImpl0} : (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.integer<8>,!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>

    %precond = "func.call"(%absres, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %postcond = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>,!transfer.integer<8>) -> i1, sym_name = "counterConcat1"} : () -> ()

"func.func"() ({
  ^bb0(%absres: !transfer.abs_value<[!transfer.integer<8>]>, %arg0: !transfer.abs_value<[!transfer.integer<8>]>, %inst: !transfer.integer<8>, %inst1: !transfer.integer<8>, %operand: !transfer.integer<8>):
    %concrete_res0 = "transfer.concat"(%operand, %inst):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %concrete_res1 = "transfer.concat"(%operand, %inst1):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>

    %precond = "func.call"(%absres, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %postcond = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>,!transfer.integer<8>) -> i1,other_operand=[4], abs_input=1,sym_name = "precision_counterConcat1"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>]>, %op0: !transfer.integer<8>, %op1: !transfer.integer<8>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>]>) -> !transfer.integer<8>
    %const0 = "transfer.constant"(%op0){value=0:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %bitwidth0 = "transfer.get_bit_width"(%op0): (!transfer.integer<8>) -> !transfer.integer<8>
    %bitwidth1 = "transfer.get_bit_width"(%op1): (!transfer.integer<8>) -> !transfer.integer<8>

    %result_0 = "transfer.extract"(%arg0_0, %bitwidth1, %const0) : (!transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_0) : (!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>]>,!transfer.integer<8>,!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>]>,
   sym_name = "ConcatImpl1", applied_to=["comb.concat"],
   operationNo=1, CPPCLASS=["circt::comb::ConcatOp"],is_forward=false, precision_util="precision_counterConcat1", soundness_counterexample="counterConcat1"} : () -> ()

}) {"builtin.NEED_VERIFY"=[["XOR","XORImpl"]]}: () -> ()
