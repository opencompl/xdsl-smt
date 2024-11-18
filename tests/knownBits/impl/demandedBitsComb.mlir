"builtin.module"() ({
"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %result = "transfer.cmp"(%arg0_0, %arg0_0){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "getConstraint"} : () -> ()

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
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %inst: !transfer.integer):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %neg_inst = "transfer.neg"(%inst) : (!transfer.integer) -> !transfer.integer
    %or1 = "transfer.or"(%neg_inst,%arg0_0): (!transfer.integer,!transfer.integer)->!transfer.integer
    %or2 = "transfer.or"(%inst,%arg0_1): (!transfer.integer,!transfer.integer)->!transfer.integer
    %cmp1="transfer.cmp"(%or1,%neg_inst){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    %cmp2="transfer.cmp"(%or2,%inst){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    %result="arith.andi"(%cmp1,%cmp2):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1, sym_name = "inKnownBits"} : () -> ()

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
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1, sym_name = "counterXor"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>):
    "func.return"(%arg0) : (!transfer.abs_value<[!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>, sym_name = "XORImpl", applied_to=["comb.xor"], CPPCLASS=["circt::comb::XorOp"], soundness_counterexample="counterXor"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %inst: !transfer.integer, %inst1: !transfer.integer, %operand: !transfer.integer, %op0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %op1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %precond = "func.call"(%arg0, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %concrete_res0 = "transfer.xor"(%inst,%operand):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %concrete_res1 = "transfer.xor"(%inst1,%operand):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %absres =  "func.call"(%arg0, %op0, %op1) {callee = @AndImpl0} : (!transfer.abs_value<[!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>
    %postcond = "func.call"(%absres, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer,!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "counterAnd0"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %op0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %op1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %op0_constraint = "func.call"(%op0) {callee = @isValidKnownBit} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1
    %op1_constraint = "func.call"(%op1) {callee = @isValidKnownBit} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1
    %result="arith.andi"(%op0_constraint,%op1_constraint):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "AbsAndConstraint"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %op0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %op1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %op0_0 = "transfer.get"(%op0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op0_1 = "transfer.get"(%op0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_1 = "transfer.get"(%op1) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %neg_op1_0 = "transfer.neg"(%op1_0) : (!transfer.integer) -> !transfer.integer
    %result_1 = "transfer.and"(%neg_op1_0, %arg0_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_1) : (!transfer.integer) -> !transfer.abs_value<[!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "AndImpl0", applied_to=["comb.and"], CPPCLASS=["circt::comb::AndOp"],abs_op_constraint="AbdAndConstraint", soundness_counterexample="counterAnd0"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %inst: !transfer.integer, %inst1: !transfer.integer, %operand: !transfer.integer):
    %precond = "func.call"(%arg0, %inst, %inst1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %concrete_res0 = "transfer.and"(%operand,%inst):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %concrete_res1 = "transfer.and"(%operand,%inst1):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %absres =  "func.call"(%arg0,%op0, %op1) {callee = @AndImpl1} : (!transfer.abs_value<[!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>
    %postcond = "func.call"(%absres, %concrete_res0, %concrete_res1) {callee = @inSameEq} : (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %precondtrue = "arith.cmpi"(%precond, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %postcondfalse = "arith.cmpi"(%postcond, %const0) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %result="arith.andi"(%precondtrue,%postcondfalse):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>, !transfer.integer, !transfer.integer,!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "counterAnd1"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer]>, %op0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %op1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer]>) -> !transfer.integer
    %op0_0 = "transfer.get"(%op0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %op1_0 = "transfer.get"(%op1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %neg_op1_0 = "transfer.neg"(%op1_0) : (!transfer.integer) -> !transfer.integer
    %and_1 = "transfer.and"(%neg_op1_0, %op0_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %neg_and = "transfer.neg"(%and_1) : (!transfer.integer) -> !transfer.integer
    %result_1 = "transfer.and"(%neg_and, %arg0_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_1) : (!transfer.integer) -> !transfer.abs_value<[!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer]>, sym_name = "AndImpl1", applied_to=["comb.and"], CPPCLASS=["circt::comb::AndOp"],abs_op_constraint="AbdAndConstraint", soundness_counterexample="counterAnd1"} : () -> ()



}) {"builtin.NEED_VERIFY"=[["XOR","XORImpl"]]}: () -> ()
