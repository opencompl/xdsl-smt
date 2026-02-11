"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %andi = "transfer.and"(%arg0_0, %arg0_1) : (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.cmp"(%andi, %const0){predicate=0:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> i1, sym_name = "getConstraint"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %inst: !transfer.integer<8>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %neg_inst = "transfer.neg"(%inst) : (!transfer.integer<8>) -> !transfer.integer<8>
    %or1 = "transfer.or"(%neg_inst,%arg0_0): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %or2 = "transfer.or"(%inst,%arg0_1): (!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %cmp1="transfer.cmp"(%or1,%neg_inst){predicate=0:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    %cmp2="transfer.cmp"(%or2,%inst){predicate=0:i64}:(!transfer.integer<8>,!transfer.integer<8>)->i1
    %result="arith.andi"(%cmp1,%cmp2):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>) -> i1, sym_name = "getInstanceConstraint"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer<8>, %arg1: !transfer.integer<8>):
    %0 = "transfer.or"(%arg0, %arg1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    "func.return"(%0) : (!transfer.integer<8>) -> ()
  }) {function_type = (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>, sym_name = "OR"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %result_0 = "transfer.and"(%arg0_0, %arg1_0) : (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.or"(%arg0_1, %arg1_1) : (!transfer.integer<8>,!transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "ORImpl", applied_to=["arith.ori"], CPPCLASS=["mlir::arith::OrIOp"]} : () -> ()
    "func.func"() ({
  ^bb0(%arg0: !transfer.integer<8>, %arg1: !transfer.integer<8>):
    %0 = "transfer.and"(%arg0, %arg1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    "func.return"(%0) : (!transfer.integer<8>) -> ()
  }) {function_type = (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>, sym_name = "AND"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %result_0 = "transfer.or"(%arg0_0, %arg1_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.and"(%arg0_1, %arg1_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "ANDImpl", applied_to=["arith.andi"], CPPCLASS=["mlir::arith::AndIOp"]} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer<8>, %arg1: !transfer.integer<8>):
    %0 = "transfer.xor"(%arg0, %arg1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    "func.return"(%0) : (!transfer.integer<8>) -> ()
  }) {function_type = (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>, sym_name = "XOR"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %and_00 = "transfer.and" (%arg0_0, %arg1_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %and_11 = "transfer.and"(%arg0_1, %arg1_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %and_01 = "transfer.and" (%arg0_0, %arg1_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %and_10 = "transfer.and"(%arg0_1, %arg1_0) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result_0 = "transfer.or" (%and_00, %and_11) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result_1 = "transfer.or" (%and_01, %and_10) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer<8>, !transfer.integer<8>) -> !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "XORImpl", applied_to=["arith.xori"], CPPCLASS=["mlir::arith::XOrIOp"]} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %result = "transfer.neg"(%arg0_0) : (!transfer.integer<8>) -> !transfer.integer<8>
    "func.return"(%result) : (!transfer.integer<8>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>, sym_name = "getMaxValue"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    "func.return"(%arg0_1) : (!transfer.integer<8>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>, sym_name = "getMinValue"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %result = "transfer.countr_one" (%arg0_0) : (!transfer.integer<8>) -> !transfer.integer<8>
    "func.return"(%result) : (!transfer.integer<8>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>, sym_name = "countMinTrailingZeros"} : () -> ()
    "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %result = "transfer.countr_one" (%arg0_1) : (!transfer.integer<8>) -> !transfer.integer<8>
    "func.return"(%result) : (!transfer.integer<8>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>, sym_name = "countMinTrailingOnes"} : () -> ()
  "func.func"() ({
  ^bb0(%lhs: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %rhs: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %carryZero:!transfer.integer<8>, %carryOne:!transfer.integer<8>):
    %lhs0 ="transfer.get"(%lhs){index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %lhs1 ="transfer.get"(%lhs){index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %rhs0 ="transfer.get"(%rhs){index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %rhs1 ="transfer.get"(%rhs){index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %one="transfer.constant"(%lhs0){value=1:index}:(!transfer.integer<8>)->!transfer.integer<8>
    %negCarryZero="transfer.sub"(%one,%carryZero):(!transfer.integer<8>,!transfer.integer<8>)->!transfer.integer<8>
    %lhsMax = "func.call"(%lhs) {callee = @getMaxValue} : (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %lhsMin = "func.call"(%lhs) {callee = @getMinValue} : (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %rhsMax = "func.call"(%rhs) {callee = @getMaxValue} : (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %rhsMin = "func.call"(%rhs) {callee = @getMinValue} : (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
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
    %result="transfer.make"(%knownZero,%knownOne):(!transfer.integer<8>,!transfer.integer<8>)->!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "computeForAddCarry"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer<8>, %arg1: !transfer.integer<8>):
    %0 = "transfer.add"(%arg0, %arg1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    "func.return"(%0) : (!transfer.integer<8>) -> ()
  }) {function_type = (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>, sym_name = "ADD"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %one="transfer.constant"(%arg1_0){value=1:index}:(!transfer.integer<8>)->!transfer.integer<8>
    %zero="transfer.constant"(%arg1_0){value=0:index}:(!transfer.integer<8>)->!transfer.integer<8>
    %result = "func.call"(%arg0,%arg1,%one,%zero){callee=@computeForAddCarry}:(!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !transfer.integer<8>, !transfer.integer<8>) -> !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "ADDImpl", applied_to=["arith.addi"], CPPCLASS=["mlir::arith::AddIOp"]} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer<8>, %arg1: !transfer.integer<8>):
    %0 = "transfer.sub"(%arg0, %arg1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    "func.return"(%0) : (!transfer.integer<8>) -> ()
  }) {function_type = (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>, sym_name = "SUB"} : () -> ()
    "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %newRhs="transfer.make"(%arg1_1,%arg1_0):(!transfer.integer<8>,!transfer.integer<8>)->!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    %one="transfer.constant"(%arg1_0){value=1:index}:(!transfer.integer<8>)->!transfer.integer<8>
    %zero="transfer.constant"(%arg1_1){value=0:index}:(!transfer.integer<8>)->!transfer.integer<8>
    %result = "func.call"(%arg0,%newRhs,%zero,%one){callee=@computeForAddCarry}:(!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!transfer.integer<8>, !transfer.integer<8>) -> !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "SUBImpl", applied_to=["arith.subi"], CPPCLASS=["mlir::arith::SubIOp"]} : () -> ()
"func.func"() ({
  ^bb0(%arg0: !transfer.integer<8>, %arg1: !transfer.integer<8>):
    %0 = "transfer.mul"(%arg0, %arg1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    "func.return"(%0) : (!transfer.integer<8>) -> ()
  }) {function_type = (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>, sym_name = "MUL"} : () -> ()
   "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, %arg1: !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>):
    %arg0Max = "func.call"(%arg0) {callee = @getMaxValue} : (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1Max = "func.call"(%arg1) {callee = @getMaxValue} : (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %umaxResult = "transfer.mul"(%arg0Max, %arg1Max) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %umaxResultOverflow = "transfer.umul_overflow"(%arg0Max, %arg1Max) : (!transfer.integer<8>, !transfer.integer<8>) -> i1
    %zero = "transfer.constant"(%arg0Max){value=0:index} : (!transfer.integer<8>) -> !transfer.integer<8>
    %umaxResult_cnt_l_zero = "transfer.countl_zero" (%umaxResult) : (!transfer.integer<8>) -> !transfer.integer<8>
    %leadZ = "arith.select" (%umaxResultOverflow, %zero, %umaxResult_cnt_l_zero): (i1, !transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !transfer.integer<8>
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
    %result = "transfer.make"(%resZero, %resOne) : (!transfer.integer<8>, !transfer.integer<8>) -> !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>
    "func.return"(%result) : (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>,!abs_value<[!transfer.integer<8>,!transfer.integer<8>]>) -> !abs_value<[!transfer.integer<8>,!transfer.integer<8>]>, sym_name = "MULImpl", applied_to=["arith.muli"], CPPCLASS=["mlir::arith::MulIOp"]} : () -> ()
}) {"builtin.NEED_VERIFY"=[["MUL","MULImpl"],["OR","ORImpl"],["AND","ANDImpl"],["XOR","XORImpl"],["ADD","ADDImpl"],["SUB","SUBImpl"]]}: () -> ()
