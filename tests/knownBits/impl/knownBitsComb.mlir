"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %andi = "transfer.and"(%arg0_0, %arg0_1) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer) -> !transfer.integer
    %result = "transfer.cmp"(%andi, %const0){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "getConstraint"} : () -> ()
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
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1, sym_name = "getInstanceConstraint"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.or"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "OR"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result_0 = "transfer.and"(%arg0_0, %arg1_0) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %result_1 = "transfer.or"(%arg0_1, %arg1_1) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "ORImpl", applied_to=["comb.or"], CPPCLASS=["circt::comb::OrOp"]} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result_0 = "transfer.and"(%arg0_0, %arg1_0) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %result_1 = "transfer.and"(%arg0_1, %arg1_1) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "intersection"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %add_res = "transfer.add"(%arg0_0, %arg0_1) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %all_ones = "transfer.get_all_ones"(%arg0_1) : (!transfer.integer) -> !transfer.integer
    %cmp_res = "transfer.cmp"(%add_res,%all_ones){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    "func.return"(%cmp_res) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "isConstant"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    "func.return"(%arg0_1) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer, sym_name = "getConstant"} : () -> ()
    "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.and"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "AND"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result_0 = "transfer.or"(%arg0_0, %arg1_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result_1 = "transfer.and"(%arg0_1, %arg1_1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "ANDImpl", applied_to=["comb.and"], CPPCLASS=["circt::comb::AndOp"]} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.xor"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "XOR"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %and_00 = "transfer.and" (%arg0_0, %arg1_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %and_11 = "transfer.and"(%arg0_1, %arg1_1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %and_01 = "transfer.and" (%arg0_0, %arg1_1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %and_10 = "transfer.and"(%arg0_1, %arg1_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result_0 = "transfer.or" (%and_00, %and_11) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result_1 = "transfer.or" (%and_01, %and_10) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "XORImpl", applied_to=["comb.xor"], CPPCLASS=["circt::comb::XorOp"]} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result = "transfer.neg"(%arg0_0) : (!transfer.integer) -> !transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer, sym_name = "getMaxValue"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    "func.return"(%arg0_1) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer, sym_name = "getMinValue"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result = "transfer.countr_one" (%arg0_0) : (!transfer.integer) -> !transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer, sym_name = "countMinTrailingZeros"} : () -> ()
    "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result = "transfer.countr_one" (%arg0_1) : (!transfer.integer) -> !transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer, sym_name = "countMinTrailingOnes"} : () -> ()
  "func.func"() ({
  ^bb0(%lhs: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %rhs: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %carryZero:!transfer.integer, %carryOne:!transfer.integer):
    %lhs0 ="transfer.get"(%lhs){index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %lhs1 ="transfer.get"(%lhs){index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %rhs0 ="transfer.get"(%rhs){index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %rhs1 ="transfer.get"(%rhs){index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %one="transfer.constant"(%lhs0){value=1:index}:(!transfer.integer)->!transfer.integer
    %negCarryZero="transfer.sub"(%one,%carryZero):(!transfer.integer,!transfer.integer)->!transfer.integer
    %lhsMax = "func.call"(%lhs) {callee = @getMaxValue} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %lhsMin = "func.call"(%lhs) {callee = @getMinValue} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %rhsMax = "func.call"(%rhs) {callee = @getMaxValue} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %rhsMin = "func.call"(%rhs) {callee = @getMinValue} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %possibleSumZeroTmp = "transfer.add" (%lhsMax,%rhsMax):(!transfer.integer,!transfer.integer) -> !transfer.integer
    %possibleSumZero="transfer.add"(%possibleSumZeroTmp,%negCarryZero): (!transfer.integer,!transfer.integer) -> !transfer.integer
    %possibleSumOneTmp = "transfer.add" (%lhsMin,%rhsMin): (!transfer.integer,!transfer.integer) -> !transfer.integer
    %possibleSumOne="transfer.add"(%possibleSumOneTmp,%carryOne):(!transfer.integer,!transfer.integer) -> !transfer.integer
    %carryKnownZeroTmp0="transfer.xor"(%possibleSumZero,%lhs0):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %carryKnownZeroTmp1="transfer.xor"(%carryKnownZeroTmp0,%rhs0):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %carryKnownZero="transfer.neg"(%carryKnownZeroTmp1):(!transfer.integer)->!transfer.integer
    %carryKnownOneTmp="transfer.xor"(%possibleSumOne,%lhs1):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %carryKnownOne="transfer.xor"(%carryKnownOneTmp,%rhs1):(!transfer.integer,!transfer.integer) ->!transfer.integer
    %lhsKnownUnion="transfer.or"(%lhs0,%lhs1):(!transfer.integer,!transfer.integer)->!transfer.integer
    %rhsKnownUnion="transfer.or"(%rhs0,%rhs1):(!transfer.integer,!transfer.integer)->!transfer.integer
    %carryKnownUnion="transfer.or"(%carryKnownZero,%carryKnownOne):(!transfer.integer,!transfer.integer)->!transfer.integer
    %knownTmp="transfer.and"(%lhsKnownUnion,%rhsKnownUnion):(!transfer.integer,!transfer.integer)->!transfer.integer
    %known="transfer.and"(%knownTmp,%carryKnownUnion):(!transfer.integer,!transfer.integer)->!transfer.integer
    %knownZeroTmp="transfer.neg"(%possibleSumZero):(!transfer.integer)->!transfer.integer
    %knownZero="transfer.and"(%knownZeroTmp,%known):(!transfer.integer,!transfer.integer)->!transfer.integer
    %knownOne="transfer.and"(%possibleSumOne,%known):(!transfer.integer,!transfer.integer)->!transfer.integer
    %result="transfer.make"(%knownZero,%knownOne):(!transfer.integer,!transfer.integer)->!transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "computeForAddCarry"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.add"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "ADD"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %one="transfer.constant"(%arg1_0){value=1:index}:(!transfer.integer)->!transfer.integer
    %zero="transfer.constant"(%arg1_0){value=0:index}:(!transfer.integer)->!transfer.integer
    %result = "func.call"(%arg0,%arg1,%one,%zero){callee=@computeForAddCarry}:(!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "ADDImpl", applied_to=["comb.add"], CPPCLASS=["circt::comb::AddOp"]} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.sub"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "SUB"} : () -> ()
    "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %newRhs="transfer.make"(%arg1_1,%arg1_0):(!transfer.integer,!transfer.integer)->!transfer.abs_value<[!transfer.integer,!transfer.integer]>
    %one="transfer.constant"(%arg1_0){value=1:index}:(!transfer.integer)->!transfer.integer
    %zero="transfer.constant"(%arg1_1){value=0:index}:(!transfer.integer)->!transfer.integer
    %result = "func.call"(%arg0,%newRhs,%zero,%one){callee=@computeForAddCarry}:(!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "SUBImpl", applied_to=["comb.sub"], CPPCLASS=["circt::comb::SubOp"]} : () -> ()
"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.mul"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "MUL"} : () -> ()
   "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0Max = "func.call"(%arg0) {callee = @getMaxValue} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1Max = "func.call"(%arg1) {callee = @getMaxValue} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %umaxResult = "transfer.mul"(%arg0Max, %arg1Max) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %umaxResultOverflow = "transfer.umul_overflow"(%arg0Max, %arg1Max) : (!transfer.integer, !transfer.integer) -> i1
    %zero = "transfer.constant"(%arg0Max){value=0:index} : (!transfer.integer) -> !transfer.integer
    %umaxResult_cnt_l_zero = "transfer.countl_zero" (%umaxResult) : (!transfer.integer) -> !transfer.integer
    %leadZ = "transfer.select" (%umaxResultOverflow, %zero, %umaxResult_cnt_l_zero): (i1, !transfer.integer, !transfer.integer) -> !transfer.integer
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %lhs_union = "transfer.or"(%arg0_0, %arg0_1): (!transfer.integer,!transfer.integer)->!transfer.integer
    %rhs_union = "transfer.or"(%arg1_0, %arg1_1): (!transfer.integer,!transfer.integer)->!transfer.integer
    %trailBitsKnown0 = "transfer.countr_one"(%lhs_union): (!transfer.integer) -> !transfer.integer
    %trailBitsKnown1 = "transfer.countr_one"(%rhs_union): (!transfer.integer) -> !transfer.integer
    %trailZero0 = "transfer.countr_one"(%arg0_0): (!transfer.integer) -> !transfer.integer
    %trailZero1 = "transfer.countr_one"(%arg1_0): (!transfer.integer) -> !transfer.integer
    %trailZ = "transfer.add"(%trailZero0, %trailZero1): (!transfer.integer,!transfer.integer) -> !transfer.integer
    %smallestOperand_arg0 = "transfer.sub"(%trailBitsKnown0, %trailZero0): (!transfer.integer,!transfer.integer)->!transfer.integer
    %smallestOperand_arg1 = "transfer.sub"(%trailBitsKnown1, %trailZero1): (!transfer.integer,!transfer.integer)->!transfer.integer
    %smallestOperand = "transfer.umin"(%smallestOperand_arg0, %smallestOperand_arg1): (!transfer.integer,!transfer.integer)->!transfer.integer
    %resultBitsKnown_arg0="transfer.add"(%smallestOperand, %trailZ): (!transfer.integer,!transfer.integer)->!transfer.integer
    %bitwidth = "transfer.get_bit_width"(%arg0_0): (!transfer.integer) -> !transfer.integer
    %resultBitsKnown = "transfer.umin"(%resultBitsKnown_arg0,%bitwidth): (!transfer.integer,!transfer.integer)->!transfer.integer
    %bottomKnown_arg0 = "transfer.get_low_bits"(%arg0_1, %trailBitsKnown0): (!transfer.integer,!transfer.integer)->!transfer.integer
    %bottomKnown_arg1 = "transfer.get_low_bits"(%arg1_1, %trailBitsKnown1): (!transfer.integer,!transfer.integer)->!transfer.integer
    %bottomKnown = "transfer.mul"(%bottomKnown_arg0, %bottomKnown_arg1): (!transfer.integer,!transfer.integer)->!transfer.integer
    %bottomKnown_neg="transfer.neg"(%bottomKnown): (!transfer.integer) -> !transfer.integer
    %resZerotmp2="transfer.get_low_bits"(%bottomKnown_neg, %resultBitsKnown): (!transfer.integer,!transfer.integer)->!transfer.integer
    %resZerotmp = "transfer.set_high_bits"(%zero, %leadZ): (!transfer.integer,!transfer.integer)->!transfer.integer
    %resZero = "transfer.or"(%resZerotmp, %resZerotmp2): (!transfer.integer,!transfer.integer)->!transfer.integer
    %resOne="transfer.get_low_bits"(%bottomKnown, %resultBitsKnown): (!transfer.integer,!transfer.integer)->!transfer.integer
    %result = "transfer.make"(%resZero, %resOne) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "MULImpl", applied_to=["comb.mul"], CPPCLASS=["circt::comb::MulOp"]} : () -> ()
"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.concat"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "CONCAT"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result_0 = "transfer.concat"(%arg0_0, %arg1_0) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %result_1 = "transfer.concat"(%arg0_1, %arg1_1) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "CONCATImpl", applied_to=["comb.concat"], CPPCLASS=["circt::comb::ConcatOp"], induction=true} : () -> ()
    "func.func"() ({
^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer, %arg2: !transfer.integer):
    %0 = "transfer.extract"(%arg0, %arg1, %arg2) : (!transfer.integer, !transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "EXTRACT"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>,%arg2: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_val = "func.call"(%arg1) {callee = @getConstant} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg2_val = "func.call"(%arg2) {callee = @getConstant} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result_0 = "transfer.extract"(%arg0_0, %arg1_val, %arg2_val) : (!transfer.integer, !transfer.integer, !transfer.integer) -> !transfer.integer
    %result_1 = "transfer.extract"(%arg0_1, %arg1_val, %arg2_val) : (!transfer.integer, !transfer.integer, !transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "EXTRACTImpl", applied_to=["comb.extract"], CPPCLASS=["circt::comb::ExtractOp"]} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.integer):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result_0_tmp = "transfer.shl"(%arg0_0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result_0 = "transfer.set_low_bits"(%result_0_tmp, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result_1 = "transfer.shl"(%arg0_1, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "shiftByConst"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %const0 = "transfer.constant"(%arg0_0) {value=0:index}:(!transfer.integer)->!transfer.integer
    %result_0 = "transfer.cmp"(%arg0_0, %const0) {predicate=0:i64}: (!transfer.integer, !transfer.integer) -> i1
    %result_1 = "transfer.cmp"(%arg0_1, %const0) {predicate=0:i64}: (!transfer.integer, !transfer.integer) -> i1
    %result = "arith.andi"(%result_0, %result_1):(i1,i1) -> i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "isUnknown"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_min = "func.call"(%arg1) {callee = @getMinValue} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result_0 = "transfer.set_low_bits"(%arg0_0, %arg1_min): (!transfer.integer,!transfer.integer)->!transfer.integer
    %result = "transfer.make"(%result_0, %arg0_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "shlLHSFastPath"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_0_rones = "transfer.countr_one"(%arg0_0): (!transfer.integer)->!transfer.integer
    %arg1_min = "func.call"(%arg1) {callee = @getMinValue} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result_0 = "transfer.set_low_bits"(%arg0_0, %arg1_min): (!transfer.integer,!transfer.integer)->!transfer.integer
    %result = "transfer.make"(%result_0, %arg0_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "shlRHSFastPath"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %bitwidth = "transfer.get_bit_width"(%arg0): (!transfer.integer) -> !transfer.integer
    %const0 = "transfer.constant"(%arg1) {value=0:index}:(!transfer.integer)->!transfer.integer
    %ge0 = "transfer.cmp"(%const0, %arg1) {predicate=7:i64}: (!transfer.integer, !transfer.integer) -> i1
    %ltSize = "transfer.cmp"(%arg1, %bitwidth) {predicate=6:i64}: (!transfer.integer, !transfer.integer) -> i1
    %check = "arith.andi"(%ge0, %ltSize) : (i1, i1) -> i1
    "func.return"(%check) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "shl_constraint"} : () -> ()


  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.shl"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "SHL",op_constraint="shl_constraint"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer) -> !transfer.integer
    %const1 = "transfer.constant"(%arg0_0){value=1:index} : (!transfer.integer) -> !transfer.integer
    %bitwidth = "transfer.get_bit_width"(%arg0_0): (!transfer.integer) -> !transfer.integer
    %result_tmp_0 = "transfer.get_all_ones"(%arg0_0): (!transfer.integer) -> !transfer.integer
    %result_tmp_1 = "transfer.get_all_ones"(%arg0_0): (!transfer.integer) -> !transfer.integer
    %result_tmp = "transfer.make"(%result_tmp_0, %result_tmp_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    %for_res = "transfer.const_range_for"(%const0, %bitwidth, %const1, %result_tmp) ({
      ^bb0(%ind: !transfer.integer, %tmp: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
        %tmp_0 = "transfer.get"(%tmp) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
        %tmp_1 = "transfer.get"(%tmp) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
        %isValidShift = "func.call"(%arg1, %ind) {callee = @getInstanceConstraint} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.integer) -> i1
        %tmp_shift = "func.call"(%arg0, %ind) {callee = @shiftByConst} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
        %intersection_tmp = "func.call"(%tmp, %tmp_shift) {callee = @intersection} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
        %intersection_tmp_0 = "transfer.get"(%intersection_tmp) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
        %intersection_tmp_1 = "transfer.get"(%intersection_tmp) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
        %tmp_res_0 = "transfer.select"(%isValidShift, %intersection_tmp_0, %tmp_0):(i1,!transfer.integer,!transfer.integer)->!transfer.integer
        %tmp_res_1 = "transfer.select"(%isValidShift, %intersection_tmp_1, %tmp_1):(i1,!transfer.integer,!transfer.integer)->!transfer.integer
        %tmp_for_res = "transfer.make"(%tmp_res_0, %tmp_res_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
        "transfer.next_loop"(%tmp_for_res) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
    }) : (!transfer.integer, !transfer.integer, !transfer.integer, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    %for_res_0 = "transfer.get"(%for_res) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %for_res_1 = "transfer.get"(%for_res) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %conflict = "transfer.intersects"(%for_res_1, %for_res_0): (!transfer.integer, !transfer.integer) -> i1
    %result_0 = "transfer.select"(%conflict, %const0, %for_res_0):(i1,!transfer.integer,!transfer.integer)->!transfer.integer
    %result_1 = "transfer.select"(%conflict, %const0, %for_res_1):(i1,!transfer.integer,!transfer.integer)->!transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "SHLImpl", applied_to=["comb.shl"], CPPCLASS=["circt::comb::ShlOp"],op_constraint="shl_constraint"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.ashr"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "ASHR"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%const0, %const0) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "SHRSImpl", applied_to=["comb.shrs"], CPPCLASS=["circt::comb::ShrSOp"]} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.lshr"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "LSHR"} : () -> ()
"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%const0, %const0) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "SHRUImpl", applied_to=["comb.shru"], CPPCLASS=["circt::comb::ShrUOp"]} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.cmp"(%arg0, %arg1) {predicate=0:i64}: (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "eq"} : () -> ()
"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_const = "func.call"(%arg0) {callee = @isConstant} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1
    %arg1_const = "func.call"(%arg1) {callee = @isConstant} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1
    %constCheck = "arith.andi"(%arg0_const, %arg1_const) : (i1, i1) -> i1
    %arg0_val = "func.call"(%arg0) {callee = @getConstant} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_val = "func.call"(%arg1) {callee = @getConstant} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %res1_1 = "transfer.cmp"(%arg0_val, %arg1_val){predicate=0:i64}: (!transfer.integer, !transfer.integer) -> i1
    %res1_0 = "arith.xori"(%res1_1, %const1) : (i1, i1) -> i1

    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %cond1 = "transfer.intersects"(%arg0_1, %arg1_0): (!transfer.integer, !transfer.integer) -> i1
    %cond2 = "transfer.intersects"(%arg0_0, %arg1_1): (!transfer.integer, !transfer.integer) -> i1
    %cond =  "arith.ori"(%cond1, %cond2) : (i1, i1) -> i1

    %result1_0 = "arith.select"(%cond, %const1, %const0):(i1,i1,i1) -> i1
    %result_0_i1 = "arith.select"(%constCheck, %res1_0, %result1_0):(i1,i1,i1) -> i1
    %result_1_i1 = "arith.select"(%constCheck, %res1_1, %const0):(i1,i1,i1) -> i1
    %result_0 = "transfer.fromArith"(%result_0_i1, %arg0_0):(i1,!transfer.integer) -> !transfer.integer
    %result_1 = "transfer.fromArith"(%result_1_i1, %arg0_0):(i1,!transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "EQImpl"} : () -> ()


  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.cmp"(%arg0, %arg1) {predicate=1:i64}: (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "ne"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %eqRes = "func.call"(%arg0,%arg1) {callee = @EQImpl} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>
    %eqRes_0 = "transfer.get"(%eqRes) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %const0 = "transfer.constant"(%eqRes_0){value=0:index} : (!transfer.integer) -> !transfer.integer
    %eqRes_1 = "transfer.get"(%eqRes) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %eqConst = "func.call"(%eqRes) {callee = @isConstant} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1

    %res_0 = "transfer.select"(%eqConst, %eqRes_1, %const0):(i1,!transfer.integer,!transfer.integer)->!transfer.integer
    %res_1 = "transfer.select"(%eqConst, %eqRes_0, %const0):(i1,!transfer.integer,!transfer.integer)->!transfer.integer
    %result = "transfer.make"(%res_0, %res_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "NEImpl"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.cmp"(%arg0, %arg1) {predicate=2:i64}: (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "slt"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %result = "transfer.make"(%const0, %const0) : (i1, i1) -> !transfer.abs_value<[i1,i1]>
    "func.return"(%result) : (!transfer.abs_value<[i1,i1]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[i1,i1]>, sym_name = "SLTImpl"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.cmp"(%arg0, %arg1) {predicate=3:i64}: (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "sle"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %result = "transfer.make"(%const0, %const0) : (i1, i1) -> !transfer.abs_value<[i1,i1]>
    "func.return"(%result) : (!transfer.abs_value<[i1,i1]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[i1,i1]>, sym_name = "SLEImpl"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.cmp"(%arg0, %arg1) {predicate=4:i64}: (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "sgt"} : () -> ()
"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %result = "transfer.make"(%const0, %const0) : (i1, i1) -> !transfer.abs_value<[i1,i1]>
    "func.return"(%result) : (!transfer.abs_value<[i1,i1]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[i1,i1]>, sym_name = "SGTImpl"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.cmp"(%arg0, %arg1) {predicate=5:i64}: (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "sge"} : () -> ()
"func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %result = "transfer.make"(%const0, %const0) : (i1, i1) -> !transfer.abs_value<[i1,i1]>
    "func.return"(%result) : (!transfer.abs_value<[i1,i1]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[i1,i1]>, sym_name = "SGEImpl"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.cmp"(%arg0, %arg1) {predicate=6:i64}: (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "ult"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %result = "transfer.make"(%const0, %const0) : (i1, i1) -> !transfer.abs_value<[i1,i1]>
    "func.return"(%result) : (!transfer.abs_value<[i1,i1]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[i1,i1]>, sym_name = "ULTImpl"} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.cmp"(%arg0, %arg1) {predicate=7:i64}: (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "ule"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %result = "transfer.make"(%const0, %const0) : (i1, i1) -> !transfer.abs_value<[i1,i1]>
    "func.return"(%result) : (!transfer.abs_value<[i1,i1]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[i1,i1]>, sym_name = "ULEImpl"} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.cmp"(%arg0, %arg1) {predicate=8:i64}: (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "ugt"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %result = "transfer.make"(%const0, %const0) : (i1, i1) -> !transfer.abs_value<[i1,i1]>
    "func.return"(%result) : (!transfer.abs_value<[i1,i1]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[i1,i1]>, sym_name = "UGTImpl"} : () -> ()



"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.cmp"(%arg0, %arg1) {predicate=9:i64}: (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "uge"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %const0 = "arith.constant"() {value=0:i1}: () -> i1
    %result = "transfer.make"(%const0, %const0) : (i1, i1) -> !transfer.abs_value<[i1,i1]>
    "func.return"(%result) : (!transfer.abs_value<[i1,i1]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[i1,i1]>, sym_name = "UGEImpl"} : () -> ()

"func.func"() ({
  ^bb0(%cond :i1, %arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.select"(%cond, %arg0, %arg1) : (i1, !transfer.integer, !transfer.integer) ->!transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (i1, !transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "MUX"} : () -> ()
"func.func"() ({
  ^bb0(%cond: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %cond_0 = "transfer.get"(%cond) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer

    %cond_const = "func.call"(%cond) {callee = @isConstant} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1
    %cond_val = "func.call"(%cond) {callee = @getConstant} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %const1 = "transfer.constant"(%cond_0){value=1:index} : (!transfer.integer) -> !transfer.integer
    %cond_eq_1 = "transfer.cmp"(%cond_val, %const1){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    %cond_res_0 = "transfer.select"(%cond_eq_1, %arg0_0, %arg1_0): (i1, !transfer.integer,!transfer.integer)->!transfer.integer
    %cond_res_1 = "transfer.select"(%cond_eq_1, %arg0_1, %arg1_1): (i1, !transfer.integer,!transfer.integer)->!transfer.integer

    %intersection_res = "func.call"(%arg0, %arg1) {callee = @intersection} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    %intersection_0 = "transfer.get"(%intersection_res) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %intersection_1 = "transfer.get"(%intersection_res) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer

    %result_0 = "transfer.select"(%cond_const, %cond_res_0, %intersection_0): (i1, !transfer.integer,!transfer.integer)->!transfer.integer
    %result_1 = "transfer.select"(%cond_const, %cond_res_1, %intersection_1): (i1, !transfer.integer,!transfer.integer)->!transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "MUXImpl", applied_to=["comb.mux"], CPPCLASS=["circt::comb::MuxOp"]} : () -> ()


"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.repeat"(%arg0, %arg1): (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "REPEAT"} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_val = "func.call"(%arg1) {callee = @getConstant} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result_0 = "transfer.repeat"(%arg0_0, %arg1_val): (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result_1 = "transfer.repeat"(%arg0_1, %arg1_val): (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "REPEATImpl", applied_to=["comb.replicate"], CPPCLASS=["circt::comb::ReplicateOp"]} : () -> ()



}) {"builtin.NEED_VERIFY"=[["MUL","MULImpl"],["OR","ORImpl"],["AND","ANDImpl"],["XOR","XORImpl"],["ADD","ADDImpl"],["SUB","SUBImpl"],["MUX","MUXImpl"],["SHL","SHLImpl"]]}: () -> ()
