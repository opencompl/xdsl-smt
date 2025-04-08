"builtin.module"() ({

 "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg00 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %const0 = "transfer.constant"(%arg00){value=0:index} : (!transfer.integer) -> !transfer.integer
    %result = "transfer.make"(%const0, %const0) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "getTop"} : () -> ()

   "func.func"() ({
  ^bb0(%arg0: !transfer.abs_value<[!transfer.integer,!transfer.integer]>, %arg1: !transfer.abs_value<[!transfer.integer,!transfer.integer]>):
    %arg00 = "transfer.get"(%arg0) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg01 = "transfer.get"(%arg0) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg10 = "transfer.get"(%arg1) {index=0:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %arg11 = "transfer.get"(%arg1) {index=1:index}: (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.integer
    %and0 = "transfer.or"(%arg00,%arg10): (!transfer.integer,!transfer.integer)->!transfer.integer
    %and1 = "transfer.or"(%arg01,%arg11): (!transfer.integer,!transfer.integer)->!transfer.integer
    %result = "transfer.make"(%and0, %and1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>
    "func.return"(%result) : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>, !transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "meet"} : () -> ()

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
    "func.return"(%arg0) : (!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> ()
  }) {function_type = (!transfer.abs_value<[!transfer.integer,!transfer.integer]>,!transfer.abs_value<[!transfer.integer,!transfer.integer]>) -> !transfer.abs_value<[!transfer.integer,!transfer.integer]>, sym_name = "AddImpl", applied_to=["comb.add"], CPPCLASS=["circt::comb::AddOp"], is_forward=true} : () -> ()

  func.func @part_solution_1_body(%arg0 : !transfer.abs_value<[!transfer.integer, !transfer.integer]>, %arg1 : !transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>  attributes {applied_to = ["comb.add"], CPPCLASS = ["circt::comb::AddOp"], is_forward = true}{
  %0 = "transfer.get"(%arg0) {index = 0 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %1 = "transfer.get"(%arg0) {index = 1 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %2 = "transfer.get"(%arg1) {index = 0 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %3 = "transfer.get"(%arg1) {index = 1 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %4 = "transfer.constant"(%3) {value = 1 : index} : (!transfer.integer) -> !transfer.integer
  %5 = "transfer.countr_zero"(%0) : (!transfer.integer) -> !transfer.integer
  %6 = "transfer.and"(%3, %1) : (!transfer.integer, !transfer.integer) -> !transfer.integer
  %7 = "transfer.countr_one"(%2) : (!transfer.integer) -> !transfer.integer
  %8 = "transfer.cmp"(%3, %5) {predicate = 0 : index} : (!transfer.integer, !transfer.integer) -> i1
  %9 = "transfer.add"(%0, %2) : (!transfer.integer, !transfer.integer) -> !transfer.integer
  %10 = "transfer.add"(%9, %4) : (!transfer.integer, !transfer.integer) -> !transfer.integer
  %11 = "transfer.select"(%8, %10, %3) : (i1, !transfer.integer, !transfer.integer) -> !transfer.integer
  %12 = "transfer.shl"(%11, %11) : (!transfer.integer, !transfer.integer) -> !transfer.integer
  %13 = "transfer.or"(%7, %6) : (!transfer.integer, !transfer.integer) -> !transfer.integer
  %14 = "transfer.countr_one"(%12) : (!transfer.integer) -> !transfer.integer
  %15 = "transfer.and"(%13, %10) : (!transfer.integer, !transfer.integer) -> !transfer.integer
  %16 = "transfer.make"(%15, %14) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>
  func.return %16 : !transfer.abs_value<[!transfer.integer, !transfer.integer]>
}

func.func @part_solution_2_cond(%0 : !transfer.abs_value<[!transfer.integer, !transfer.integer]>, %1 : !transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> i1 {
  %2 = "transfer.get"(%0) {index = 0 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %3 = "transfer.get"(%1) {index = 0 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %4 = "transfer.get"(%1) {index = 1 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %5 = "transfer.get_all_ones"(%4) : (!transfer.integer) -> !transfer.integer
  %6 = "transfer.or"(%3, %2) : (!transfer.integer, !transfer.integer) -> !transfer.integer
  %7 = "transfer.cmp"(%5, %6) {predicate = 7 : index} : (!transfer.integer, !transfer.integer) -> i1
  func.return %7 : i1
}

func.func @part_solution_2_body(%arg0 : !transfer.abs_value<[!transfer.integer, !transfer.integer]>, %arg1 : !transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>  attributes {applied_to = ["comb.add"], CPPCLASS = ["circt::comb::AddOp"], is_forward = true}{
  %0 = "transfer.get"(%arg0) {index = 0 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %1 = "transfer.get"(%arg0) {index = 1 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %2 = "transfer.get"(%arg1) {index = 0 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %3 = "transfer.get"(%arg1) {index = 1 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
  %4 = "transfer.xor"(%1, %3) : (!transfer.integer, !transfer.integer) -> !transfer.integer
  %5 = "transfer.and"(%0, %2) : (!transfer.integer, !transfer.integer) -> !transfer.integer
  %6 = "transfer.make"(%5, %4) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>
  func.return %6 : !transfer.abs_value<[!transfer.integer, !transfer.integer]>
}


}): () -> ()
