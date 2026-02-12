"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %andi = "transfer.and"(%arg0_0, %arg0_1) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %const0 = "transfer.constant"(%arg0_0){value=0:index} : (!transfer.integer) -> !transfer.integer
    %result = "transfer.cmp"(%andi, %const0){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!abs_value<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "getConstraint"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer,!transfer.integer]>, %inst: !transfer.integer):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %neg_inst = "transfer.neg"(%inst) : (!transfer.integer) -> !transfer.integer
    %or1 = "transfer.or"(%neg_inst,%arg0_0): (!transfer.integer,!transfer.integer)->!transfer.integer
    %or2 = "transfer.or"(%inst,%arg0_1): (!transfer.integer,!transfer.integer)->!transfer.integer
    %cmp1="transfer.cmp"(%or1,%neg_inst){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    %cmp2="transfer.cmp"(%or2,%inst){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    %result="arith.andi"(%cmp1,%cmp2):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!abs_value<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1, sym_name = "getInstanceConstraint"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.or"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "OR"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result_0 = "transfer.and"(%arg0_0, %arg1_0) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %result_1 = "transfer.or"(%arg0_1, %arg1_1) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer,!transfer.integer]>,!abs_value<[!transfer.integer,!transfer.integer]>) -> !abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "ORImpl", applied_to=["arith.ori"], CPPCLASS=["mlir::arith::OrIOp"]} : () -> ()
    "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.and"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "AND"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result_0 = "transfer.or"(%arg0_0, %arg1_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result_1 = "transfer.and"(%arg0_1, %arg1_1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer,!transfer.integer]>,!abs_value<[!transfer.integer,!transfer.integer]>) -> !abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "ANDImpl", applied_to=["arith.andi"], CPPCLASS=["mlir::arith::AndIOp"]} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.xor"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "XOR"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %and_00 = "transfer.and" (%arg0_0, %arg1_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %and_11 = "transfer.and"(%arg0_1, %arg1_1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %and_01 = "transfer.and" (%arg0_0, %arg1_1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %and_10 = "transfer.and"(%arg0_1, %arg1_0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result_0 = "transfer.or" (%and_00, %and_11) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result_1 = "transfer.or" (%and_01, %and_10) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer,!transfer.integer]>,!abs_value<[!transfer.integer,!transfer.integer]>) -> !abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "XORImpl", applied_to=["arith.xori"], CPPCLASS=["mlir::arith::XOrIOp"]} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result = "transfer.neg"(%arg0_0) : (!transfer.integer) -> !transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer, sym_name = "getMaxValue"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    "func.return"(%arg0_1) : (!transfer.integer) -> ()
  }) {function_type = (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer, sym_name = "getMinValue"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result = "transfer.countr_one" (%arg0_0) : (!transfer.integer) -> !transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer, sym_name = "countMinTrailingZeros"} : () -> ()
    "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result = "transfer.countr_one" (%arg0_1) : (!transfer.integer) -> !transfer.integer
    "func.return"(%result) : (!transfer.integer) -> ()
  }) {function_type = (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer, sym_name = "countMinTrailingOnes"} : () -> ()
  "func.func"() ({
  ^bb0(%lhs: !abs_value<[!transfer.integer,!transfer.integer]>, %rhs: !abs_value<[!transfer.integer,!transfer.integer]>, %carryZero:!transfer.integer, %carryOne:!transfer.integer):
    %lhs0 ="transfer.get"(%lhs){index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %lhs1 ="transfer.get"(%lhs){index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %rhs0 ="transfer.get"(%rhs){index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %rhs1 ="transfer.get"(%rhs){index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %one="transfer.constant"(%lhs0){value=1:index}:(!transfer.integer)->!transfer.integer
    %negCarryZero="transfer.sub"(%one,%carryZero):(!transfer.integer,!transfer.integer)->!transfer.integer
    %lhsMax = "func.call"(%lhs) {callee = @getMaxValue} : (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %lhsMin = "func.call"(%lhs) {callee = @getMinValue} : (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %rhsMax = "func.call"(%rhs) {callee = @getMaxValue} : (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %rhsMin = "func.call"(%rhs) {callee = @getMinValue} : (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
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
    %result="transfer.make"(%knownZero,%knownOne):(!transfer.integer,!transfer.integer)->!abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer,!transfer.integer]>, !abs_value<[!transfer.integer,!transfer.integer]>, !transfer.integer, !transfer.integer) -> !abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "computeForAddCarry"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.add"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "ADD"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !abs_value<[!transfer.integer,!transfer.integer]>):
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %one="transfer.constant"(%arg1_0){value=1:index}:(!transfer.integer)->!transfer.integer
    %zero="transfer.constant"(%arg1_0){value=0:index}:(!transfer.integer)->!transfer.integer
    %result = "func.call"(%arg0,%arg1,%one,%zero){callee=@computeForAddCarry}:(!abs_value<[!transfer.integer,!transfer.integer]>, !abs_value<[!transfer.integer,!transfer.integer]>, !transfer.integer, !transfer.integer) -> !abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer,!transfer.integer]>,!abs_value<[!transfer.integer,!transfer.integer]>) -> !abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "ADDImpl", applied_to=["arith.addi"], CPPCLASS=["mlir::arith::AddIOp"]} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.sub"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "SUB"} : () -> ()
    "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !abs_value<[!transfer.integer,!transfer.integer]>):
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %newRhs="transfer.make"(%arg1_1,%arg1_0):(!transfer.integer,!transfer.integer)->!abs_value<[!transfer.integer,!transfer.integer]>
    %one="transfer.constant"(%arg1_0){value=1:index}:(!transfer.integer)->!transfer.integer
    %zero="transfer.constant"(%arg1_1){value=0:index}:(!transfer.integer)->!transfer.integer
    %result = "func.call"(%arg0,%newRhs,%zero,%one){callee=@computeForAddCarry}:(!abs_value<[!transfer.integer,!transfer.integer]>, !abs_value<[!transfer.integer,!transfer.integer]>,!transfer.integer, !transfer.integer) -> !abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer,!transfer.integer]>,!abs_value<[!transfer.integer,!transfer.integer]>) -> !abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "SUBImpl", applied_to=["arith.subi"], CPPCLASS=["mlir::arith::SubIOp"]} : () -> ()
"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.mul"(%arg0, %arg1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    "func.return"(%0) : (!transfer.integer) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> !transfer.integer, sym_name = "MUL"} : () -> ()
   "func.func"() ({
  ^bb0(%arg0: !abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !abs_value<[!transfer.integer,!transfer.integer]>):
    %arg0Max = "func.call"(%arg0) {callee = @getMaxValue} : (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1Max = "func.call"(%arg1) {callee = @getMaxValue} : (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %umaxResult = "transfer.mul"(%arg0Max, %arg1Max) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %umaxResultOverflow = "transfer.umul_overflow"(%arg0Max, %arg1Max) : (!transfer.integer, !transfer.integer) -> i1
    %zero = "transfer.constant"(%arg0Max){value=0:index} : (!transfer.integer) -> !transfer.integer
    %umaxResult_cnt_l_zero = "transfer.countl_zero" (%umaxResult) : (!transfer.integer) -> !transfer.integer
    %leadZ = "arith.select" (%umaxResultOverflow, %zero, %umaxResult_cnt_l_zero): (i1, !transfer.integer, !transfer.integer) -> !transfer.integer
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
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
    %result = "transfer.make"(%resZero, %resOne) : (!transfer.integer, !transfer.integer) -> !abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!abs_value<[!transfer.integer,!transfer.integer]>,!abs_value<[!transfer.integer,!transfer.integer]>) -> !abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "MULImpl", applied_to=["arith.muli"], CPPCLASS=["mlir::arith::MulIOp"]} : () -> ()
}) {"builtin.NEED_VERIFY"=[["MUL","MULImpl"],["OR","ORImpl"],["AND","ANDImpl"],["XOR","XORImpl"],["ADD","ADDImpl"],["SUB","SUBImpl"]]}: () -> ()
