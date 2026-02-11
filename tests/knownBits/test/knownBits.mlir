"builtin.module"() ({

  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %andi = "transfer.and"(%arg0_0, %arg0_1) : (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.cmp"(%andi, %const0){predicate=0:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1, sym_name = "getConstraint"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %inst: !transfer.integer<8>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %neg_inst = "transfer.neg"(%inst) : (!transfer.integer<8>) -> !transfer.integer<8>
    %or1 = "transfer.or"(%neg_inst,%arg0_0): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %or2 = "transfer.or"(%inst,%arg0_1): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %cmp1="transfer.cmp"(%or1,%neg_inst){predicate=0:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    %cmp2="transfer.cmp"(%or2,%inst){predicate=0:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    %result="arith.andi"(%cmp1,%cmp2):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1, sym_name = "getInstanceConstraint"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %result_0 = "transfer.and"(%arg0_0, %arg1_0) : (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.and"(%arg0_1, %arg1_1) : (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "intersection"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %add_res = "transfer.add"(%arg0_0, %arg0_1) : (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %all_ones = "transfer.get_all_ones"(%arg0_1) : (!transfer.integer<8>) -> !transfer.integer<8>
    %cmp_res = "transfer.cmp"(%add_res,%all_ones){predicate=0:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    "func.return"(%cmp_res) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1, sym_name = "isConstant"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    "func.return"(%arg0_1) : (!transfer.integer<8>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>, sym_name = "getConstant"} : () -> ()

 "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[i1,i1]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[i1,i1]>) -> i1
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[i1,i1]>) -> i1

    %arg0_0_arith ="transfer.add_poison"(%arg0_0): (i1) -> i1
    %arg0_1_arith ="transfer.add_poison"(%arg0_1): (i1) -> i1
    %add_res = "arith.addi"(%arg0_0_arith, %arg0_1_arith) : (i1,i1) -> i1
    %all_ones = "arith.constant"() {"value" = 1 : i1} : () -> i1
    %cmp_res = "arith.cmpi"(%add_res,%all_ones) {"predicate" = 0 : i64} : (i1,i1) -> i1
    "func.return"(%cmp_res) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[i1,i1]>) -> i1, sym_name = "isConstant_i1"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[i1,i1]>):
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[i1,i1]>) -> i1
    %arg0_1_arith ="transfer.add_poison"(%arg0_1): (i1) -> i1
    "func.return"(%arg0_1_arith) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[i1,i1]>) -> i1, sym_name = "getConstant_i1"} : () -> ()


"func.func"() ({
  ^bb0(%cond: !transfer.abs_value<[i1,i1]>, %arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):

  %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %cond_0 = "transfer.get"(%cond) {index=0:index}: (!transfer.abs_value<[i1,i1]>) -> i1

    %cond_const = "func.call"(%cond) {callee = @isConstant_i1} : (!transfer.abs_value<[i1,i1]>) -> i1
    %cond_val = "func.call"(%cond) {callee = @getConstant_i1} : (!transfer.abs_value<[i1,i1]>) -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %cond_eq_1 = "arith.cmpi"(%cond_val, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %cond_res_0 = "transfer.select"(%cond_eq_1, %arg0_0, %arg1_0): (i1, !transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %cond_res_1 = "transfer.select"(%cond_eq_1, %arg0_1, %arg1_1): (i1, !transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>

    %intersection_res = "func.call"(%arg0, %arg1) {callee = @intersection} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    %intersection_0 = "transfer.get"(%intersection_res) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %intersection_1 = "transfer.get"(%intersection_res) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>

    %result_0 = "transfer.select"(%cond_const, %cond_res_0, %intersection_0): (i1, !transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %result_1 = "transfer.select"(%cond_const, %cond_res_1, %intersection_1): (i1, !transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()

  }) {function_type = (!transfer.abs_value<[i1,i1]>, !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "MUXImpl", applied_to=["comb.mux"], CPPCLASS=["circt::comb::MuxOp"], is_forward=true} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %result_0 = "transfer.or"(%arg0_0, %arg1_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.and"(%arg0_1, %arg1_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "ANDImpl", applied_to=["comb.and"], CPPCLASS=["circt::comb::AndOp"], is_forward=true} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %and_00 = "transfer.and" (%arg0_0, %arg1_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %and_11 = "transfer.and"(%arg0_1, %arg1_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %and_01 = "transfer.and" (%arg0_0, %arg1_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %and_10 = "transfer.and"(%arg0_1, %arg1_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result_0 = "transfer.or" (%and_00, %and_11) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.or" (%and_01, %and_10) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "XORImpl", applied_to=["comb.xor"], CPPCLASS=["circt::comb::XorOp"], is_forward=true} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %result_0 = "transfer.and"(%arg0_0, %arg1_0) : (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.or"(%arg0_1, %arg1_1) : (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "ORImpl", applied_to=["comb.or"], CPPCLASS=["circt::comb::OrOp"], is_forward=true} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %result = "transfer.neg"(%arg0_0) : (!transfer.integer<8>) -> !transfer.integer<8>
    "func.return"(%result) : (!transfer.integer<8>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>, sym_name = "getMaxValue"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    "func.return"(%arg0_1) : (!transfer.integer<8>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>, sym_name = "getMinValue"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %result = "transfer.countr_one" (%arg0_0) : (!transfer.integer<8>) -> !transfer.integer<8>
    "func.return"(%result) : (!transfer.integer<8>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>, sym_name = "countMinTrailingZeros"} : () -> ()
    "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %result = "transfer.countr_one" (%arg0_1) : (!transfer.integer<8>) -> !transfer.integer<8>
    "func.return"(%result) : (!transfer.integer<8>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>, sym_name = "countMinTrailingOnes"} : () -> ()

"func.func"() ({
  ^bb0(%lhs: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %rhs: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %carryZero:!transfer.integer<8>, %carryOne:!transfer.integer<8>):
    %lhs0 ="transfer.get"(%lhs){index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %lhs1 ="transfer.get"(%lhs){index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %rhs0 ="transfer.get"(%rhs){index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %rhs1 ="transfer.get"(%rhs){index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %one="transfer.constant"(%lhs0){value=1:index}:(!transfer.integer<8>)->!transfer.integer<8>
    %negCarryZero="transfer.sub"(%one,%carryZero):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %lhsMax = "func.call"(%lhs) {callee = @getMaxValue} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %lhsMin = "func.call"(%lhs) {callee = @getMinValue} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %rhsMax = "func.call"(%rhs) {callee = @getMaxValue} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %rhsMin = "func.call"(%rhs) {callee = @getMinValue} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %possibleSumZeroTmp = "transfer.add" (%lhsMax,%rhsMax):(!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %possibleSumZero="transfer.add"(%possibleSumZeroTmp,%negCarryZero): (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %possibleSumOneTmp = "transfer.add" (%lhsMin,%rhsMin): (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %possibleSumOne="transfer.add"(%possibleSumOneTmp,%carryOne):(!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %carryKnownZeroTmp0="transfer.xor"(%possibleSumZero,%lhs0):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %carryKnownZeroTmp1="transfer.xor"(%carryKnownZeroTmp0,%rhs0):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %carryKnownZero="transfer.neg"(%carryKnownZeroTmp1):(!transfer.integer<8>)->!transfer.integer<8>
    %carryKnownOneTmp="transfer.xor"(%possibleSumOne,%lhs1):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %carryKnownOne="transfer.xor"(%carryKnownOneTmp,%rhs1):(!transfer.integer<8>,!transfer.integer<8>) ->!transfer.integer<8>
    %lhsKnownUnion="transfer.or"(%lhs0,%lhs1):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %rhsKnownUnion="transfer.or"(%rhs0,%rhs1):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %carryKnownUnion="transfer.or"(%carryKnownZero,%carryKnownOne):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %knownTmp="transfer.and"(%lhsKnownUnion,%rhsKnownUnion):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %known="transfer.and"(%knownTmp,%carryKnownUnion):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %knownZeroTmp="transfer.neg"(%possibleSumZero):(!transfer.integer<8>)->!transfer.integer<8>
    %knownZero="transfer.and"(%knownZeroTmp,%known):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %knownOne="transfer.and"(%possibleSumOne,%known):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %result="transfer.make"(%knownZero,%knownOne):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "computeForAddCarry"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %one="transfer.constant"(%arg1_0){value=1:index}:(!transfer.integer<8>)->!transfer.integer<8>
    %zero="transfer.constant"(%arg1_0){value=0:index}:(!transfer.integer<8>)->!transfer.integer<8>
    %result = "func.call"(%arg0,%arg1,%one,%zero){callee=@computeForAddCarry}:(!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "ADDImpl", applied_to=["comb.add"], CPPCLASS=["circt::comb::AddOp"],is_forward=true} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %newRhs="transfer.make"(%arg1_1,%arg1_0):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    %one="transfer.constant"(%arg1_0){value=1:index}:(!transfer.integer<8>)->!transfer.integer<8>
    %zero="transfer.constant"(%arg1_1){value=0:index}:(!transfer.integer<8>)->!transfer.integer<8>
    %result = "func.call"(%arg0,%newRhs,%zero,%one){callee=@computeForAddCarry}:(!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "SUBImpl", applied_to=["comb.sub"], CPPCLASS=["circt::comb::SubOp"], is_forward=true} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %result_0 = "transfer.concat"(%arg0_0, %arg1_0) : (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.concat"(%arg0_1, %arg1_1) : (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "CONCATImpl", applied_to=["comb.concat"], CPPCLASS=["circt::comb::ConcatOp"],is_forward=true, induction=true} : () -> ()

"func.func"() ({
^bb0(%arg0: !transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>, %len:!transfer.integer<8>,%low_bit :!transfer.integer<8>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %bitwidth = "transfer.get_bit_width"(%arg0_0): (!transfer.integer<8>) -> !transfer.integer<8>
    %add_res = "transfer.add"(%len, %low_bit) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.cmp"(%add_res, %bitwidth){predicate=7:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1

    %low_bit_res = "transfer.cmp"(%low_bit, %bitwidth){predicate=6:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1

    %const1 = "transfer.constant"(%len){value=1:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %len_ge_1 = "transfer.cmp"(%len, %const1){predicate=9:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    %result_2="arith.andi"(%result_1,%len_ge_1):(i1,i1)->i1
    %result="arith.andi"(%result_2,%low_bit_res):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.tuple<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.integer<8>,!transfer.integer<8>) -> i1,
  sym_name = "EXTRACTAttrConstraint"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %len:!transfer.integer<8> ,%low_bit:!transfer.integer<8> ):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %result_0 = "transfer.extract"(%arg0_0, %len, %low_bit) : (!transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.extract"(%arg0_1, %len, %low_bit) : (!transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.integer<8>,!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "EXTRACTImpl", applied_to=["comb.extract"], CPPCLASS=["circt::comb::ExtractOp"],is_forward=true, int_attr=[1,2],int_attr_constraint="EXTRACTAttrConstraint", replace_int_attr=true } : () -> ()


 "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0Max = "func.call"(%arg0) {callee = @getMaxValue} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1Max = "func.call"(%arg1) {callee = @getMaxValue} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %umaxResult = "transfer.mul"(%arg0Max, %arg1Max) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %umaxResultOverflow = "transfer.umul_overflow"(%arg0Max, %arg1Max) : (!transfer.integer<8>, !transfer.integer<8>) -> i1
    %zero = "transfer.constant"(%arg0Max){value=0:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %umaxResult_cnt_l_zero = "transfer.countl_zero" (%umaxResult) : (!transfer.integer<8>) -> !transfer.integer<8>
    %leadZ = "transfer.select" (%umaxResultOverflow, %zero, %umaxResult_cnt_l_zero): (i1, !transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %lhs_union = "transfer.or"(%arg0_0, %arg0_1): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %rhs_union = "transfer.or"(%arg1_0, %arg1_1): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %trailBitsKnown0 = "transfer.countr_one"(%lhs_union): (!transfer.integer<8>) -> !transfer.integer<8>
    %trailBitsKnown1 = "transfer.countr_one"(%rhs_union): (!transfer.integer<8>) -> !transfer.integer<8>
    %trailZero0 = "transfer.countr_one"(%arg0_0): (!transfer.integer<8>) -> !transfer.integer<8>
    %trailZero1 = "transfer.countr_one"(%arg1_0): (!transfer.integer<8>) -> !transfer.integer<8>
    %trailZ = "transfer.add"(%trailZero0, %trailZero1): (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %smallestOperand_arg0 = "transfer.sub"(%trailBitsKnown0, %trailZero0): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %smallestOperand_arg1 = "transfer.sub"(%trailBitsKnown1, %trailZero1): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %smallestOperand = "transfer.umin"(%smallestOperand_arg0, %smallestOperand_arg1): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %resultBitsKnown_arg0="transfer.add"(%smallestOperand, %trailZ): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %bitwidth = "transfer.get_bit_width"(%arg0_0): (!transfer.integer<8>) -> !transfer.integer<8>
    %resultBitsKnown = "transfer.umin"(%resultBitsKnown_arg0,%bitwidth): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %bottomKnown_arg0 = "transfer.get_low_bits"(%arg0_1, %trailBitsKnown0): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %bottomKnown_arg1 = "transfer.get_low_bits"(%arg1_1, %trailBitsKnown1): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %bottomKnown = "transfer.mul"(%bottomKnown_arg0, %bottomKnown_arg1): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %bottomKnown_neg="transfer.neg"(%bottomKnown): (!transfer.integer<8>) -> !transfer.integer<8>
    %resZerotmp2="transfer.get_low_bits"(%bottomKnown_neg, %resultBitsKnown): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %resZerotmp = "transfer.set_high_bits"(%zero, %leadZ): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %resZero = "transfer.or"(%resZerotmp, %resZerotmp2): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %resOne="transfer.get_low_bits"(%bottomKnown, %resultBitsKnown): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %result = "transfer.make"(%resZero, %resOne) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,is_forward=true, sym_name = "MULImpl", applied_to=["comb.mul"], CPPCLASS=["circt::comb::MulOp"]} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !transfer.integer<8>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %result_0_tmp = "transfer.shl"(%arg0_0, %arg1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result_0 = "transfer.set_low_bits"(%result_0_tmp, %arg1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.shl"(%arg0_1, %arg1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "shiftByConst"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.integer<8>, %arg1: !transfer.integer<8>):
    %bitwidth = "transfer.get_bit_width"(%arg0): (!transfer.integer<8>) -> !transfer.integer<8>
    %const0 = "transfer.constant"(%arg1) {value=0:index}:(!transfer.integer<8>)->!transfer.integer<8>
    %ge0 = "transfer.cmp"(%const0, %arg1) {predicate=7:i64}: (!transfer.integer<8>, !transfer.integer<8>) -> i1
    %ltSize = "transfer.cmp"(%arg1, %bitwidth) {predicate=6:i64}: (!transfer.integer<8>, !transfer.integer<8>) -> i1
    %check = "arith.andi"(%ge0, %ltSize) : (i1, i1) -> i1
    "func.return"(%check) : (i1) -> ()
  }) {function_type = (!transfer.integer<8>, !transfer.integer<8>) -> i1, sym_name = "shl_constraint"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %const1 = "transfer.constant"(%arg0_0){value=1:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %bitwidth = "transfer.get_bit_width"(%arg0_0): (!transfer.integer<8>) -> !transfer.integer<8>
    %result_tmp_0 = "transfer.get_all_ones"(%arg0_0): (!transfer.integer<8>) -> !transfer.integer<8>
    %result_tmp_1 = "transfer.get_all_ones"(%arg0_0): (!transfer.integer<8>) -> !transfer.integer<8>
    %result_tmp = "transfer.make"(%result_tmp_0, %result_tmp_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    %for_res = "transfer.const_range_for"(%const0, %bitwidth, %const1, %result_tmp) ({
      ^bb0(%ind: !transfer.integer<8>, %tmp: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
        %tmp_0 = "transfer.get"(%tmp) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
        %tmp_1 = "transfer.get"(%tmp) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
        %isValidShift = "func.call"(%arg1, %ind) {callee = @getInstanceConstraint} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.integer<8>) -> i1
        %tmp_shift = "func.call"(%arg0, %ind) {callee = @shiftByConst} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
        %intersection_tmp = "func.call"(%tmp, %tmp_shift) {callee = @intersection} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
        %intersection_tmp_0 = "transfer.get"(%intersection_tmp) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
        %intersection_tmp_1 = "transfer.get"(%intersection_tmp) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
        %tmp_res_0 = "transfer.select"(%isValidShift, %intersection_tmp_0, %tmp_0):(i1,!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
        %tmp_res_1 = "transfer.select"(%isValidShift, %intersection_tmp_1, %tmp_1):(i1,!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
        %tmp_for_res = "transfer.make"(%tmp_res_0, %tmp_res_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
        "transfer.next_loop"(%tmp_for_res) : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
    }) : (!transfer.integer<8>, !transfer.integer<8>, !transfer.integer<8>, !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    %for_res_0 = "transfer.get"(%for_res) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %for_res_1 = "transfer.get"(%for_res) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %conflict = "transfer.intersects"(%for_res_1, %for_res_0): (!transfer.integer<8>, !transfer.integer<8>) -> i1
    %result_0 = "transfer.select"(%conflict, %const0, %for_res_0):(i1,!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %result_1 = "transfer.select"(%conflict, %const0, %for_res_1):(i1,!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "SHLImpl", applied_to=["comb.shl"], CPPCLASS=["circt::comb::ShlOp"],op_constraint="shl_constraint", is_forward=true} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_const = "func.call"(%arg0) {callee = @isConstant} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %arg1_const = "func.call"(%arg1) {callee = @isConstant} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1
    %constCheck = "arith.andi"(%arg0_const, %arg1_const) : (i1, i1) -> i1
    %arg0_val = "func.call"(%arg0) {callee = @getConstant} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_val = "func.call"(%arg1) {callee = @getConstant} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %res1_1 = "transfer.cmp"(%arg0_val, %arg1_val){predicate=0:i64}: (!transfer.integer<8>, !transfer.integer<8>) -> i1
    %res1_0 = "arith.xori"(%res1_1, %const1) : (i1, i1) -> i1

    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %cond1 = "transfer.intersects"(%arg0_1, %arg1_0): (!transfer.integer<8>, !transfer.integer<8>) -> i1
    %cond2 = "transfer.intersects"(%arg0_0, %arg1_1): (!transfer.integer<8>, !transfer.integer<8>) -> i1
    %cond =  "arith.ori"(%cond1, %cond2) : (i1, i1) -> i1

    %result1_0 = "arith.select"(%cond, %const1, %const0):(i1,i1,i1) -> i1
    %result_0_i1 = "arith.select"(%constCheck, %res1_0, %result1_0):(i1,i1,i1) -> i1
    %result_1_i1 = "arith.select"(%constCheck, %res1_1, %const0):(i1,i1,i1) -> i1
    %result_0 = "transfer.remove_poison"(%result_0_i1): (i1)->i1
    %result_1 = "transfer.remove_poison"(%result_1_i1): (i1)->i1
    %result = "transfer.make"(%result_0, %result_1) : (i1, i1) -> !transfer.abs_value<[i1,i1]>
    "func.return"(%result) : (!transfer.abs_value<[i1,i1]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[i1,i1]>,applied_to=["comb.icmp",0], sym_name = "EQImpl", is_forward=true} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.integer<8>, %arg1: !transfer.integer<8>):
    %0 = "transfer.cmp"(%arg0, %arg1) {predicate=1:i64}: (!transfer.integer<8>, !transfer.integer<8>) -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (!transfer.integer<8>, !transfer.integer<8>) -> i1, sym_name = "ne"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %eqRes = "func.call"(%arg0,%arg1) {callee = @EQImpl} : (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[i1, i1]>
    %eqRes_0_i1 = "transfer.get"(%eqRes) {index=0:index}: (!transfer.abs_value<[i1,i1]>) -> i1
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %eqRes_1_i1 = "transfer.get"(%eqRes) {index=1:index}: (!transfer.abs_value<[i1,i1]>) -> i1
    %eqConst = "func.call"(%eqRes) {callee = @isConstant_i1} : (!transfer.abs_value<[i1,i1]>) -> i1
    %eqRes_0 = "transfer.add_poison"(%eqRes_0_i1):(i1)->i1
    %eqRes_1 = "transfer.add_poison"(%eqRes_1_i1):(i1)->i1


    %res_0_i1 = "arith.select"(%eqConst, %eqRes_1, %const0):(i1,i1,i1)->i1
    %res_1_i1 = "arith.select"(%eqConst, %eqRes_0, %const0):(i1,i1,i1)->i1
    %res_0 = "transfer.remove_poison"(%res_0_i1): (i1)->i1
    %res_1 = "transfer.remove_poison"(%res_1_i1): (i1)->i1

    %result = "transfer.make"(%res_0, %res_1) : (i1, i1) -> !transfer.abs_value<[i1,i1]>
    "func.return"(%result) : (!transfer.abs_value<[i1,i1]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.abs_value<[i1,i1]>,applied_to=["comb.icmp",1], sym_name = "NEImpl", is_forward=true} : () -> ()

}) {"builtin.NEED_VERIFY"=[["MUX","MUXImpl"]]}: () -> ()
