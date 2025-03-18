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
  ^bb0(%cond: !transfer.abs_value<[i1,i1]>, %arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):

  %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %cond_0 = "transfer.get"(%cond) {index=0:index}: (!transfer.abs_value<[i1,i1]>) -> i1

    %cond_const = "func.call"(%cond) {callee = @isConstant_i1} : (!transfer.abs_value<[i1,i1]>) -> i1
    %cond_val = "func.call"(%cond) {callee = @getConstant_i1} : (!transfer.abs_value<[i1,i1]>) -> i1
    %const1 = "arith.constant"() {value=1:i1}: () -> i1
    %cond_eq_1 = "arith.cmpi"(%cond_val, %const1) {"predicate" = 0 : i64} : (i1, i1) -> i1
    %cond_res_0 = "transfer.select"(%cond_eq_1, %arg0_0, %arg1_0): (i1, !transfer.integer,!transfer.integer)->!transfer.integer
    %cond_res_1 = "transfer.select"(%cond_eq_1, %arg0_1, %arg1_1): (i1, !transfer.integer,!transfer.integer)->!transfer.integer

    %intersection_res = "func.call"(%arg0, %arg1) {callee = @intersection} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    %intersection_0 = "transfer.get"(%intersection_res) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %intersection_1 = "transfer.get"(%intersection_res) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer

    %result_0 = "transfer.select"(%cond_const, %cond_res_0, %intersection_0): (i1, !transfer.integer,!transfer.integer)->!transfer.integer
    %result_1 = "transfer.select"(%cond_const, %cond_res_1, %intersection_1): (i1, !transfer.integer,!transfer.integer)->!transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()

  }) {function_type = (!transfer.abs_value<[i1,i1]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "MUXImpl", applied_to=["comb.mux"], CPPCLASS=["circt::comb::MuxOp"], is_forward=true} : () -> ()

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
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "ANDImpl", applied_to=["comb.and"], CPPCLASS=["circt::comb::AndOp"], is_forward=true} : () -> ()


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
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "XORImpl", applied_to=["comb.xor"], CPPCLASS=["circt::comb::XorOp"], is_forward=true} : () -> ()

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
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "ORImpl", applied_to=["comb.or"], CPPCLASS=["circt::comb::OrOp"], is_forward=true} : () -> ()

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
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %one="transfer.constant"(%arg1_0){value=1:index}:(!transfer.integer)->!transfer.integer
    %zero="transfer.constant"(%arg1_0){value=0:index}:(!transfer.integer)->!transfer.integer
    %result = "func.call"(%arg0,%arg1,%one,%zero){callee=@computeForAddCarry}:(!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "ADDImpl", applied_to=["comb.add"], CPPCLASS=["circt::comb::AddOp"],is_forward=true} : () -> ()

  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg1_0 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg1_1 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %newRhs="transfer.make"(%arg1_1,%arg1_0):(!transfer.integer,!transfer.integer)->!transfer.abs_value<[!transfer.integer,!transfer.integer]>
    %one="transfer.constant"(%arg1_0){value=1:index}:(!transfer.integer)->!transfer.integer
    %zero="transfer.constant"(%arg1_1){value=0:index}:(!transfer.integer)->!transfer.integer
    %result = "func.call"(%arg0,%newRhs,%zero,%one){callee=@computeForAddCarry}:(!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "SUBImpl", applied_to=["comb.sub"], CPPCLASS=["circt::comb::SubOp"], is_forward=true} : () -> ()

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
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "CONCATImpl", applied_to=["comb.concat"], CPPCLASS=["circt::comb::ConcatOp"],is_forward=true, induction=true} : () -> ()
  
"func.func"() ({
^bb0(%arg0: !transfer.tuple<[!transfer.integer,!transfer.integer]>, %len:!transfer.integer,%low_bit :!transfer.integer):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.tuple<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %bitwidth = "transfer.get_bit_width"(%arg0_0): (!transfer.integer) -> !transfer.integer
    %add_res = "transfer.add"(%len, %low_bit) : (!transfer.integer, !transfer.integer) -> !transfer.integer
    %result_1 = "transfer.cmp"(%add_res, %bitwidth){predicate=7:i64}:(!transfer.integer,!transfer.integer)->i1

    %low_bit_res = "transfer.cmp"(%low_bit, %bitwidth){predicate=6:i64}:(!transfer.integer,!transfer.integer)->i1

    %const1 = "transfer.constant"(%len){value=1:index} : (!transfer.integer) -> !transfer.integer
    %len_ge_1 = "transfer.cmp"(%len, %const1){predicate=9:i64}:(!transfer.integer,!transfer.integer)->i1
    %result_2="arith.andi"(%result_1,%len_ge_1):(i1,i1)->i1
    %result="arith.andi"(%result_2,%low_bit_res):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.tuple<[!transfer.integer,!transfer.integer]>,!transfer.integer,!transfer.integer) -> i1,
  sym_name = "EXTRACTAttrConstraint"} : () -> ()  
  
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %len:!transfer.integer ,%low_bit:!transfer.integer ):
    %arg0_0 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg0_1 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %result_0 = "transfer.extract"(%arg0_0, %len, %low_bit) : (!transfer.integer, !transfer.integer, !transfer.integer) -> !transfer.integer
    %result_1 = "transfer.extract"(%arg0_1, %len, %low_bit) : (!transfer.integer, !transfer.integer, !transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%result_0, %result_1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.integer,!transfer.integer) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "EXTRACTImpl", applied_to=["comb.extract"], CPPCLASS=["circt::comb::ExtractOp"],is_forward=true, int_attr=[1,2],int_attr_constraint="EXTRACTAttrConstraint", replace_int_attr=true } : () -> ()


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
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %bitwidth = "transfer.get_bit_width"(%arg0): (!transfer.integer) -> !transfer.integer
    %const0 = "transfer.constant"(%arg1) {value=0:index}:(!transfer.integer)->!transfer.integer
    %ge0 = "transfer.cmp"(%const0, %arg1) {predicate=7:i64}: (!transfer.integer, !transfer.integer) -> i1
    %ltSize = "transfer.cmp"(%arg1, %bitwidth) {predicate=6:i64}: (!transfer.integer, !transfer.integer) -> i1
    %check = "arith.andi"(%ge0, %ltSize) : (i1, i1) -> i1
    "func.return"(%check) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "shl_constraint"} : () -> ()

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
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "SHLImpl", applied_to=["comb.shl"], CPPCLASS=["circt::comb::ShlOp"],op_constraint="shl_constraint", is_forward=true} : () -> ()

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
    %result_0 = "transfer.remove_poison"(%result_0_i1): (i1)->i1
    %result_1 = "transfer.remove_poison"(%result_1_i1): (i1)->i1
    %result = "transfer.make"(%result_0, %result_1) : (i1, i1) -> !transfer.abs_value<[i1,i1]>
    "func.return"(%result) : (!transfer.abs_value<[i1,i1]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[i1,i1]>,applied_to=["comb.icmp",0], sym_name = "EQImpl", is_forward=true} : () -> ()

"func.func"() ({
  ^bb0(%arg0: !transfer.integer, %arg1: !transfer.integer):
    %0 = "transfer.cmp"(%arg0, %arg1) {predicate=1:i64}: (!transfer.integer, !transfer.integer) -> i1
    "func.return"(%0) : (i1) -> ()
  }) {function_type = (!transfer.integer, !transfer.integer) -> i1, sym_name = "ne"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %eqRes = "func.call"(%arg0,%arg1) {callee = @EQImpl} : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[i1, i1]>
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
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[i1,i1]>,applied_to=["comb.icmp",1], sym_name = "NEImpl", is_forward=true} : () -> ()

}) {"builtin.NEED_VERIFY"=[["MUX","MUXImpl"]]}: () -> ()
