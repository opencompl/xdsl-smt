"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg00 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg01 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %andi = "transfer.and"(%arg00, %arg01) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %const0 = "transfer.constant"(%arg00){value=0:index} : (!transfer.integer) -> !transfer.integer
    %result = "transfer.cmp"(%andi, %const0){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> i1, sym_name = "getConstraint"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %inst: !transfer.integer):
    %arg00 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg01 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %neg_inst = "transfer.neg"(%inst) : (!transfer.integer) -> !transfer.integer
    %or1 = "transfer.or"(%neg_inst,%arg00): (!transfer.integer,!transfer.integer)->!transfer.integer
    %or2 = "transfer.or"(%inst,%arg01): (!transfer.integer,!transfer.integer)->!transfer.integer
    %cmp1="transfer.cmp"(%or1,%neg_inst){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    %cmp2="transfer.cmp"(%or2,%inst){predicate=0:i64}:(!transfer.integer,!transfer.integer)->i1
    %result="arith.andi"(%cmp1,%cmp2):(i1,i1)->i1
    "func.return"(%result) : (i1) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.integer) -> i1, sym_name = "getInstanceConstraint"} : () -> ()
  "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
     %bitw = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %carryZero="transfer.constant"(%bitw){value=1:index}:(!transfer.integer)->!transfer.integer
    %carryOne="transfer.constant"(%bitw){value=0:index}:(!transfer.integer)->!transfer.integer

    %lhs0 ="transfer.get"(%arg0){index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %lhs1 ="transfer.get"(%arg0){index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %rhs0 ="transfer.get"(%arg1){index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %rhs1 ="transfer.get"(%arg1){index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %one="transfer.constant"(%lhs0){value=1:index}:(!transfer.integer)->!transfer.integer

    %negCarryZero="transfer.sub"(%one,%carryZero):(!transfer.integer,!transfer.integer)->!transfer.integer
    // 0

    %lhsMax = "transfer.neg"(%lhs0) : (!transfer.integer) -> !transfer.integer
    // !l0
    %lhsMin = "transfer.or"(%lhs1,%lhs1):(!transfer.integer,!transfer.integer)->!transfer.integer
    // l1
    %rhsMax = "transfer.neg"(%rhs0) : (!transfer.integer) -> !transfer.integer
    // !r0
    %rhsMin = "transfer.or"(%rhs1,%rhs1):(!transfer.integer,!transfer.integer)->!transfer.integer
    // r1

    %possibleSumZeroTmp = "transfer.add" (%lhsMax,%rhsMax):(!transfer.integer,!transfer.integer) -> !transfer.integer
    // !l0 + !r0
    %possibleSumZero="transfer.add"(%possibleSumZeroTmp,%negCarryZero): (!transfer.integer,!transfer.integer) -> !transfer.integer
    // !l0 + !r0
    %possibleSumOneTmp = "transfer.add" (%lhsMin,%rhsMin): (!transfer.integer,!transfer.integer) -> !transfer.integer
    // l1 + r1
    %possibleSumOne="transfer.add"(%possibleSumOneTmp,%carryOne):(!transfer.integer,!transfer.integer) -> !transfer.integer
    // l1 + r1
    %carryKnownZeroTmp0="transfer.xor"(%possibleSumZero,%lhs0):(!transfer.integer,!transfer.integer) ->!transfer.integer
    // (!l0 + !r0) ^ l0
    %carryKnownZeroTmp1="transfer.xor"(%carryKnownZeroTmp0,%rhs0):(!transfer.integer,!transfer.integer) ->!transfer.integer
    // (!l0 + !r0) ^ l0 ^ r0
    %carryKnownZero="transfer.neg"(%carryKnownZeroTmp1):(!transfer.integer)->!transfer.integer
    // !((!l0 + !r0) ^ l0 ^ r0)
    %carryKnownOneTmp="transfer.xor"(%possibleSumOne,%lhs1):(!transfer.integer,!transfer.integer) ->!transfer.integer
    // (l1 + r1) ^ l1
    %carryKnownOne="transfer.xor"(%carryKnownOneTmp,%rhs1):(!transfer.integer,!transfer.integer) ->!transfer.integer
    // (l1 + r1) ^ l1 ^ r1
    %lhsKnownUnion="transfer.or"(%lhs0,%lhs1):(!transfer.integer,!transfer.integer)->!transfer.integer
    // l0 | l1
    %rhsKnownUnion="transfer.or"(%rhs0,%rhs1):(!transfer.integer,!transfer.integer)->!transfer.integer
    // r0 | r1
    %carryKnownUnion="transfer.or"(%carryKnownZero,%carryKnownOne):(!transfer.integer,!transfer.integer)->!transfer.integer
    // !((!l0 + !r0) ^ l0 ^ r0) | ((l1 + r1) ^ l1 ^ r1)
    %knownTmp="transfer.and"(%lhsKnownUnion,%rhsKnownUnion):(!transfer.integer,!transfer.integer)->!transfer.integer
    // (l0 | l1) & (r0 | r1)
    %known="transfer.and"(%knownTmp,%carryKnownUnion):(!transfer.integer,!transfer.integer)->!transfer.integer
    // (l0 | l1) & (r0 | r1) & (!((!l0 + !r0) ^ l0 ^ r0) | ((l1 + r1) ^ l1 ^ r1))
    %knownZeroTmp="transfer.neg"(%possibleSumZero):(!transfer.integer)->!transfer.integer
    // !(!l0 + !r0)
    %knownZero="transfer.and"(%knownZeroTmp,%known):(!transfer.integer,!transfer.integer)->!transfer.integer
    // !(!l0 + !r0) & (l0 | l1) & (r0 | r1) & (!((!l0 + !r0) ^ l0 ^ r0) | ((l1 + r1) ^ l1 ^ r1))
    %knownOne="transfer.and"(%possibleSumOne,%known):(!transfer.integer,!transfer.integer)->!transfer.integer
    // (l1 + r1) & (l0 | l1) & (r0 | r1) & (!((!l0 + !r0) ^ l0 ^ r0) | ((l1 + r1) ^ l1 ^ r1))
    %result="transfer.make"(%knownZero,%knownOne):(!transfer.integer,!transfer.integer)->!transfer.abs_value<[!transfer.integer,!transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "AddImpl", applied_to=["comb.add"], CPPCLASS=["circt::comb::AddOp"], is_forward=true} : () -> ()

}) {"builtin.NEED_VERIFY"=[["MUL","MULImpl"],["OR","ORImpl"],["AND","ANDImpl"],["XOR","XORImpl"],["ADD","ADDImpl"],["SUB","SUBImpl"],["MUX","MUXImpl"],["SHL","SHLImpl"]]}: () -> ()
