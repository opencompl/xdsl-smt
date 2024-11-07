// RUN: xdsl-smt "%s" -p dce -o "%t" -t mlir && xdsl-tv "%s" "%t" | z3 -in | filecheck %s

"builtin.module"() ({
  "func.func"() ({
    %x = "arith.constant"() {"value" = 3 : i32} : () -> i32
    %unused = "arith.constant"() {"value" = 42 : i32} : () -> i32
    "func.return"(%x) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = () -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK: unsat
