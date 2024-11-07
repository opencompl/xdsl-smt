// RUN: xdsl-smt "%s" -t=smt | z3 -in | filecheck "%s"

builtin.module {
  %0 = "smt.declare_const"() : () -> !smt.bool
  %x = "smt.declare_const"() : () -> !smt.int.int
  %y = "smt.declare_const"() : () -> !smt.int.int
  %zero = "smt.int.constant"() {"value" = 0 : i32} : () -> !smt.int.int
  %one = "smt.int.constant"() {"value" = 1 : i32} : () -> !smt.int.int
  // z1 = x + y - y
  %add_x_y = "smt.int.add"(%x, %y) : (!smt.int.int, !smt.int.int) -> !smt.int.int
  %add_x_y_sub_y = "smt.int.sub"(%add_x_y, %y) : (!smt.int.int, !smt.int.int) -> !smt.int.int
  // z2 = x + 0
  %add_zero = "smt.int.add"(%x, %one) : (!smt.int.int, !smt.int.int) -> !smt.int.int
  // for all w, z1 % w == z2 % w
  %29 = "smt.forall"() ({
  ^0(%int_max : !smt.int.int):
    %is_zero = "smt.eq"(%int_max, %zero) : (!smt.int.int, !smt.int.int) -> !smt.bool
    %not_zero = "smt.ite"(%is_zero, %one, %int_max) : (!smt.bool, !smt.int.int, !smt.int.int) -> !smt.int.int
    %31 = "smt.int.mod"(%add_x_y_sub_y, %not_zero) : (!smt.int.int, !smt.int.int) -> !smt.int.int
    %32 = "smt.int.mod"(%add_zero, %not_zero) : (!smt.int.int, !smt.int.int) -> !smt.int.int
    %33 = "smt.eq"(%31, %32) : (!smt.int.int, !smt.int.int) -> !smt.bool
    "smt.yield"(%33) : (!smt.bool) -> ()
  }) : () -> !smt.bool
  "smt.assert"(%29) : (!smt.bool) -> ()
  "smt.check_sat"() : () -> ()
}

// CHECK: unsat
