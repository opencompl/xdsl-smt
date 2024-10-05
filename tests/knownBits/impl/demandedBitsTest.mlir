"builtin.module"() ({
"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %result = "transfer.cmp"(%arg0_0, %arg0_0){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>) -> i1, sym_name = "getConstraint"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %inst: !transfer.integer):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %result = "transfer.cmp"(%arg0_0, %arg0_0){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    "func.return"(%result) : (i1) -> ()
}) {function_type = (!transfer.abs_value<[!transfer.integer]>, !transfer.integer) -> i1, sym_name = "getInstanceConstraint"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %andi = "transfer.and"(%arg0_0, %arg0_1) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer) -> !transfer.integer
    %result = "transfer.cmp"(%andi, %const0){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "isValidKnownBit"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %inst: !transfer.integer):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %neg_inst = "transfer.neg"(%inst) : (!transfer.integer) -> !transfer.integer
    %or1 = "transfer.or"(%neg_inst,%arg0_0): (!transfer.integer,!transfer.integer)->!transfer.integer
    %or2 = "transfer.or"(%inst,%arg0_1): (!transfer.integer,!transfer.integer)->!transfer.integer
    %cmp1="transfer.cmp"(%or1,%neg_inst){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    %cmp2="transfer.cmp"(%or2,%inst){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    %result="arith.andi"(%cmp1,%cmp2):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1, sym_name = "inKnownBits"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %inst: !transfer.integer, %inst1: !transfer.integer):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %eqinst="transfer.and"(%arg0_0,%inst):(!transfer.integer,!transfer.integer)->!transfer.integer
    %eqinst1="transfer.and"(%arg0_0,%inst1):(!transfer.integer,!transfer.integer)->!transfer.integer
    %eq = "transfer.cmp"(%eqinst, %eqinst1){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    "func.return"(%eq) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1, sym_name = "inSameEq"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %inst: !transfer.integer, %inst1: !transfer.integer, %operand: !transfer.integer):
    %precond = "func.call"(%arg0, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %concrete_res0 = "transfer.xor"(%inst,%operand):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %concrete_res1 = "transfer.xor"(%inst1,%operand):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %absres =  "func.call"(%arg0) {callee = @XORImpl} : (!transfer.abs_value<[!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>
    %postcond = "func.call"(%absres, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer,!transfer.integer) -> i1, sym_name = "counterXor"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>):
    "func.return"(%arg0) : (!transfer.abs_value<[!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>, sym_name = "XORImpl", operationNo=0, applied_to=["comb.xor"], is_forward=false, CPPCLASS=["circt::comb::XorOp"], soundness_counterexample="counterXor"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %inst: !transfer.integer, %inst1: !transfer.integer, %operand: !transfer.integer, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %concrete_res0 = "transfer.and"(%inst,%operand):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %concrete_res1 = "transfer.and"(%inst1,%operand):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @AndImpl0} : (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer, !transfer.integer,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "counterAnd0"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %op0_0 = "transfer.get"(%op0) {index=0:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op0_1 = "transfer.get"(%op0) {index=1:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %neg_op1_0 = "transfer.neg"(%op1_0) : (!transfer.integer) -> !transfer.integer
    %result_1 = "transfer.and"(%neg_op1_0, %arg0_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_1) : (!transfer.integer) -> !transfer.abs_value<[!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>, operationNo=0, sym_name = "AndImpl0", applied_to=["comb.and"], CPPCLASS=["circt::comb::AndOp"],is_forward=false, soundness_counterexample="counterAnd0"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %inst: !transfer.integer, %inst1: !transfer.integer, %operand: !transfer.integer, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %concrete_res0 = "transfer.and"(%operand,%inst):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %concrete_res1 = "transfer.and"(%operand,%inst1):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @AndImpl0} : (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer, !transfer.integer,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "counterAnd1"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %op0_0 = "transfer.get"(%op0) {index=0:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op0_1 = "transfer.get"(%op0) {index=1:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %neg_op1_0 = "transfer.neg"(%op1_0) : (!transfer.integer) -> !transfer.integer
    %and_neg = "transfer.and"(%neg_op1_0, %op0_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %neg_and= "transfer.neg"(%and_neg) : (!transfer.integer) -> !transfer.integer
    %result_1 = "transfer.and"(%neg_and, %arg0_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_1) : (!transfer.integer) -> !transfer.abs_value<[!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>, sym_name = "AndImpl1", operationNo=1, applied_to=["comb.and"], CPPCLASS=["circt::comb::AndOp"],is_forward=false, soundness_counterexample="counterAnd1"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %inst: !transfer.integer, %inst1: !transfer.integer, %operand: !transfer.integer, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %concrete_res0 = "transfer.or"(%inst,%operand):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %concrete_res1 = "transfer.or"(%inst1,%operand):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @OrImpl0} : (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer, !transfer.integer,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "counterOr0"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %op0_0 = "transfer.get"(%op0) {index=0:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op0_1 = "transfer.get"(%op0) {index=1:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %neg_op1_1 = "transfer.neg"(%op1_1) : (!transfer.integer) -> !transfer.integer
    %result_1 = "transfer.and"(%neg_op1_1, %arg0_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_1) : (!transfer.integer) -> !transfer.abs_value<[!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>, sym_name = "OrImpl0", applied_to=["comb.or"], operationNo=0, CPPCLASS=["circt::comb::OrOp"],is_forward=false, soundness_counterexample="counterOr0"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %inst: !transfer.integer, %inst1: !transfer.integer, %operand: !transfer.integer, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %concrete_res0 = "transfer.or"(%operand, %inst):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %concrete_res1 = "transfer.or"(%operand, %inst1):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @OrImpl1} : (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer, !transfer.integer,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "counterOr1"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %op0_0 = "transfer.get"(%op0) {index=0:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op0_1 = "transfer.get"(%op0) {index=1:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %neg_op1_1 = "transfer.neg"(%op1_1) : (!transfer.integer) -> !transfer.integer
    %and_neg = "transfer.and"(%neg_op1_1, %op0_1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %neg_and = "transfer.neg"(%and_neg) : (!transfer.integer) -> !transfer.integer
    %result_1 = "transfer.and"(%neg_and, %arg0_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_1) : (!transfer.integer) -> !transfer.abs_value<[!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>, sym_name = "OrImpl1", applied_to=["comb.or"], operationNo=1, CPPCLASS=["circt::comb::OrOp"],is_forward=false, soundness_counterexample="counterOr1"} : () -> ()

"func.func"() ({
  ^bb0(%operationNo:i1, %arg0: !transfer.abs_value<[!transfer.integer]>, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %carryZero: !transfer.integer, %carryOne: !transfer.integer):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %op0_0 = "transfer.get"(%op0) {index=0:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op0_1 = "transfer.get"(%op0) {index=1:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer

    %and_0_0 = "transfer.and"(%op0_0, %op1_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %and_1_1 = "transfer.and"(%op0_1, %op1_1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %bound = "transfer.or"(%and_0_0, %and_1_1) : (!transfer.integer, !transfer.integer) -> !transfer.integer

    %rbound = "transfer.reverse_bits"(%bound) : (!transfer.integer) -> !transfer.integer
    %rarg0_0 = "transfer.reverse_bits"(%arg0_0) : (!transfer.integer) -> !transfer.integer
    %neg_rbound = "transfer.neg"(%rbound) : (!transfer.integer) -> !transfer.integer
    %or_rarg0_0_neg = "transfer.or"(%rarg0_0, %neg_rbound) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %rprop = "transfer.add"(%rarg0_0, %or_rarg0_0_neg) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %racarry = "transfer.xor"(%rprop, %neg_rbound) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %acarry = "transfer.reverse_bits"(%racarry) : (!transfer.integer) -> !transfer.integer

    %neg_op0_0 = "transfer.neg"(%op0_0) : (!transfer.integer) -> !transfer.integer
    %neg_op0_1 = "transfer.neg"(%op0_1) : (!transfer.integer) -> !transfer.integer
    %neg_op1_0 = "transfer.neg"(%op1_0) : (!transfer.integer) -> !transfer.integer
    %neg_op1_1 = "transfer.neg"(%op1_1) : (!transfer.integer) -> !transfer.integer

    %or_0_0_neg_1 = "transfer.or"(%op0_0, %neg_op1_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %or_0_neg_0_1 = "transfer.or"(%neg_op0_0, %op1_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %or_1_0_neg_1 = "transfer.or"(%op0_1, %neg_op1_1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %or_1_neg_0_1 = "transfer.or"(%neg_op0_1, %op1_1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %neededToMaintainCarryZero ="transfer.select"(%operationNo, %or_0_0_neg_1, %or_1_0_neg_1) : (i1, !transfer.integer, !transfer.integer) ->!transfer.integer
    %neededToMaintainCarryOne ="transfer.select"(%operationNo, %or_0_neg_0_1, %or_1_neg_0_1) : (i1, !transfer.integer, !transfer.integer) ->!transfer.integer

    %one="transfer.constant"(%arg0_0){value=1:index}:(!transfer.integer)->!transfer.integer
    %negCarryZero="transfer.sub"(%one,%carryZero):(!transfer.integer,!transfer.integer)->!transfer.integer
    %possibleSumZeroTmp = "transfer.add" (%neg_op0_0,%neg_op1_0):(!transfer.integer,!transfer.integer) -> !transfer.integer
    %possibleSumZero="transfer.add"(%possibleSumZeroTmp,%negCarryZero): (!transfer.integer,!transfer.integer) -> !transfer.integer
    %neg_possibleSumZero = "transfer.neg"(%possibleSumZero) : (!transfer.integer) -> !transfer.integer
    %possibleSumOneTmp = "transfer.add" (%neg_op0_1,%neg_op1_1): (!transfer.integer,!transfer.integer) -> !transfer.integer
    %possibleSumOne="transfer.add"(%possibleSumOneTmp,%carryOne):(!transfer.integer,!transfer.integer) -> !transfer.integer


    %neededToMaintainCarry_0 = "transfer.or"(%neg_possibleSumZero, %neededToMaintainCarryZero) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %neededToMaintainCarry_1 = "transfer.or"(%possibleSumOne, %neededToMaintainCarryOne) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %neededToMaintainCarry = "transfer.and"(%neededToMaintainCarry_0, %neededToMaintainCarry_1) : (!transfer.integer, !transfer.integer) -> !transfer.integer

    %carryAnd = "transfer.and"(%acarry, %neededToMaintainCarry) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result_1 = "transfer.or"(%arg0_0, %carryAnd) : (!transfer.integer, !transfer.integer) -> !transfer.integer

    %result = "transfer.make"(%result_1) : (!transfer.integer) -> !transfer.abs_value<[!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer]>) -> ()
  }) {function_type = (i1, !transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.integer,!transfer.integer) -> !transfer.abs_value<[!transfer.integer]>, sym_name = "determineLiveOperandBitsAddCarry"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %inst: !transfer.integer, %inst1: !transfer.integer, %operand: !transfer.integer, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %concrete_res0 = "transfer.add"(%inst, %operand):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %concrete_res1 = "transfer.add"(%inst1, %operand):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @AddImpl0} : (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer, !transfer.integer,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "counterAdd0"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1

    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %transfer_const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer) -> !transfer.integer
    %transfer_const1 = "transfer.constant"(%arg0_0){value=1:index} : (!transfer.integer) -> !transfer.integer

    %result = "func.call"(%const0, %arg0, %op0, %op1, %transfer_const1, %transfer_const0) {callee = @determineLiveOperandBitsAddCarry} : (i1, !transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.integer,!transfer.integer) -> !transfer.abs_value<[!transfer.integer]>

    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>, sym_name = "AddImpl0", applied_to=["comb.add"], operationNo=0, CPPCLASS=["circt::comb::AddOp"],is_forward=false, soundness_counterexample="counterAdd0"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %inst: !transfer.integer, %inst1: !transfer.integer, %operand: !transfer.integer, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %concrete_res0 = "transfer.add"(%operand, %inst):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %concrete_res1 = "transfer.add"(%operand, %inst1):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @AddImpl1} : (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer, !transfer.integer,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "counterAdd1"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1

    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %transfer_const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer) -> !transfer.integer
    %transfer_const1 = "transfer.constant"(%arg0_0){value=1:index} : (!transfer.integer) -> !transfer.integer

    %result = "func.call"(%const1, %arg0, %op0, %op1, %transfer_const1, %transfer_const0) {callee = @determineLiveOperandBitsAddCarry} : (i1, !transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.integer,!transfer.integer) -> !transfer.abs_value<[!transfer.integer]>

    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>, sym_name = "AddImpl1", applied_to=["comb.add"], operationNo=0, CPPCLASS=["circt::comb::AddOp"],is_forward=false, soundness_counterexample="counterAdd1"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %inst: !transfer.integer, %inst1: !transfer.integer, %operand: !transfer.integer, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %concrete_res0 = "transfer.sub"(%operand, %inst):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %concrete_res1 = "transfer.sub"(%operand, %inst1):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @SubImpl0} : (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer, !transfer.integer,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "counterSub0"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1

    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %newOp="transfer.make"(%op1_1,%op1_0):(!transfer.integer,!transfer.integer)->!transfer.tuple<[!transfer.integer,!transfer.integer]>
    %transfer_const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer) -> !transfer.integer
    %transfer_const1 = "transfer.constant"(%arg0_0){value=1:index} : (!transfer.integer) -> !transfer.integer

    %result = "func.call"(%const0, %arg0, %op0, %newOp, %transfer_const0, %transfer_const1) {callee = @determineLiveOperandBitsAddCarry} : (i1, !transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.integer,!transfer.integer) -> !transfer.abs_value<[!transfer.integer]>

    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>, sym_name = "SubImpl0", applied_to=["comb.sub"], operationNo=0, CPPCLASS=["circt::comb::SubOp"],is_forward=false, soundness_counterexample="counterSub0"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %inst: !transfer.integer, %inst1: !transfer.integer, %operand: !transfer.integer, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %concrete_res0 = "transfer.sub"(%operand, %inst):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %concrete_res1 = "transfer.sub"(%operand, %inst1):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %crt_res_in_abs_res = "func.call"(%arg0, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %abs_arg =  "func.call"(%arg0, %op0, %op1) {callee = @SubImpl1} : (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>
    %crt_arg_in_abs_arg = "func.call"(%abs_arg, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%crt_arg_in_abs_arg, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%crt_res_in_abs_res, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1

    %inst_kb = "func.call"(%op0, %inst) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %inst1_kb = "func.call"(%op0, %inst1) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %operand_kb = "func.call"(%op1, %operand) {callee = @inKnownBits} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1
    %two_inst_kb = "arith.andi"(%inst_kb,%inst1_kb):(i1,i1)->i1
    %operands_kb=  "arith.andi"(%two_inst_kb,%operand_kb):(i1,i1)->i1

    %op0_kb = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op1_kb = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1
    %op_kb = "arith.andi"(%op0_kb,%op1_kb):(i1,i1)->i1

    %result_1="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    %result_2 = "arith.andi"(%result_1,%operands_kb):(i1,i1)->i1
    %result = "arith.andi"(%result_2,%op_kb):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer, !transfer.integer,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "counterSub1"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %op0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %op1: !transfer.tuple<[!transfer.integer,!transfer.integer]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1

    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %newOp="transfer.make"(%op1_1,%op1_0):(!transfer.integer,!transfer.integer)->!transfer.tuple<[!transfer.integer,!transfer.integer]>
    %transfer_const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer) -> !transfer.integer
    %transfer_const1 = "transfer.constant"(%arg0_0){value=1:index} : (!transfer.integer) -> !transfer.integer

    %result = "func.call"(%const1, %arg0, %op0, %newOp, %transfer_const0, %transfer_const1) {callee = @determineLiveOperandBitsAddCarry} : (i1, !transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.integer,!transfer.integer) -> !transfer.abs_value<[!transfer.integer]>

    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>, sym_name = "SubImpl1", applied_to=["comb.sub"], operationNo=1, CPPCLASS=["circt::comb::SubOp"],is_forward=false, soundness_counterexample="counterSub1"} : () -> ()

}) {"builtin.NEED_VERIFY"=[["XOR","XORImpl"]]}: () -> ()
