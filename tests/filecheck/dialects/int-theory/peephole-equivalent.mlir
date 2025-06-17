// RUN: xdsl-smt "%s" -t=smt | z3 -in | filecheck "%s"

builtin.module {
  %zero = "smt.int.constant"() {value = 0 : i32} : () -> !smt.int.int
  // Declaration of pow
  %pow2 = "smt.declare_fun"() {"name" = "pow2"}: () -> ((!smt.int.int) -> (!smt.int.int))
  %verify_get_zero = "smt.forall"() ({
    ^0(%x : !smt.int.int):
      %x_gt_0 = "smt.int.gt"(%x, %zero) : (!smt.int.int, !smt.int.int) -> !smt.bool
      %pow2_x = "smt.call"(%pow2, %x) : ((!smt.int.int) -> (!smt.int.int),!smt.int.int) -> (!smt.int.int)
      %pow2_x_gt_0 = "smt.int.gt"(%pow2_x, %zero) : (!smt.int.int, !smt.int.int) -> !smt.bool
      %imp = smt.implies %x_gt_0, %pow2_x_gt_0
      "smt.yield"(%imp) : (!smt.bool) -> ()
  }): () -> !smt.bool
  %verify_order = "smt.forall"() ({
    ^0(%x : !smt.int.int,%y: !smt.int.int):
      %x_gt_y = "smt.int.gt"(%x, %y) : (!smt.int.int, !smt.int.int) -> !smt.bool
      %pow2_x = "smt.call"(%pow2, %x) : ((!smt.int.int) -> (!smt.int.int),!smt.int.int) -> (!smt.int.int)
      %pow2_y = "smt.call"(%pow2, %y) : ((!smt.int.int) -> (!smt.int.int),!smt.int.int) -> (!smt.int.int)
      %pow2_x_gt_pow2_y = "smt.int.gt"(%pow2_x, %pow2_y) : (!smt.int.int, !smt.int.int) -> !smt.bool
      %imp = smt.implies %x_gt_y, %pow2_x_gt_pow2_y
      "smt.yield"(%imp) : (!smt.bool) -> ()
  }): () -> !smt.bool
  "smt.assert"(%verify_get_zero) : (!smt.bool) -> ()
  "smt.assert"(%verify_order) : (!smt.bool) -> ()
  // Declare the variables
  %a = "smt.declare_const"() : () -> !smt.int.int
  %b = "smt.declare_const"() : () -> !smt.int.int
  %w = "smt.declare_const"() : () -> !smt.int.int
  %int_max = "smt.call"(%pow2, %w) : ((!smt.int.int) -> (!smt.int.int),!smt.int.int) -> (!smt.int.int)
  // Preconditions
  %a_gt_0 = "smt.int.gt"(%a, %zero) : (!smt.int.int, !smt.int.int) -> !smt.bool
  %b_gt_0 = "smt.int.gt"(%b, %zero) : (!smt.int.int, !smt.int.int) -> !smt.bool
  %w_gt_0 = "smt.int.gt"(%w, %zero) : (!smt.int.int, !smt.int.int) -> !smt.bool
  "smt.assert"(%a_gt_0) : (!smt.bool) -> ()
  "smt.assert"(%b_gt_0) : (!smt.bool) -> ()
  "smt.assert"(%w_gt_0) : (!smt.bool) -> ()
  %a_le_max = "smt.int.le"(%a,%int_max) : (!smt.int.int, !smt.int.int) -> !smt.bool
  %b_le_max = "smt.int.le"(%b, %int_max) : (!smt.int.int, !smt.int.int) -> !smt.bool
  %w_le_max = "smt.int.le"(%w, %int_max) : (!smt.int.int, !smt.int.int) -> !smt.bool
  "smt.assert"(%a_le_max) : (!smt.bool) -> ()
  "smt.assert"(%b_le_max) : (!smt.bool) -> ()
  "smt.assert"(%w_le_max) : (!smt.bool) -> ()
  %a_mod = "smt.int.mod"(%a, %int_max) : (!smt.int.int, !smt.int.int) -> !smt.int.int
  %a_eq = "smt.eq"(%a,%a_mod) : (!smt.int.int, !smt.int.int) -> !smt.bool
  %b_mod = "smt.int.mod"(%b, %int_max) : (!smt.int.int, !smt.int.int) -> !smt.int.int
  %b_eq = "smt.eq"(%b,%b_mod) : (!smt.int.int, !smt.int.int) -> !smt.bool
  "smt.assert"(%a_eq) : (!smt.bool) -> ()
  "smt.assert"(%b_eq) : (!smt.bool) -> ()
  // The peephole optimization verification
  %a_plus_b = "smt.int.add"(%a,%b) : (!smt.int.int, !smt.int.int) -> !smt.int.int
  %a_plus_b_mod = "smt.int.mod"(%a_plus_b, %int_max) : (!smt.int.int, !smt.int.int) -> !smt.int.int
  %a_plus_b_min_a = "smt.int.sub"(%a_plus_b_mod,%a) : (!smt.int.int, !smt.int.int) -> !smt.int.int
  %a_plus_b_min_a_plus_max = "smt.int.add"(%a_plus_b_min_a, %int_max) : (!smt.int.int, !smt.int.int) -> !smt.int.int
  %norm_res = "smt.int.mod"(%a_plus_b_min_a_plus_max, %int_max) : (!smt.int.int, !smt.int.int) -> !smt.int.int
  %distinct = "smt.distinct"(%norm_res,%b) : (!smt.int.int, !smt.int.int) -> !smt.bool
  "smt.assert"(%distinct) : (!smt.bool) -> ()
  //
  "smt.check_sat"() : () -> ()
}

// CHECK: unsat
