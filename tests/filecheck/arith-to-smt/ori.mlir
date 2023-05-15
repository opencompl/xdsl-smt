// RUN: xdsl-smt.py %s -p=arith-to-smt,canonicalize-smt -t=smt | filecheck %s
// RUN: xdsl-smt.py %s -p=arith-to-smt,canonicalize-smt -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %r = "arith.ori"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      (define-fun tmp ((x (_ BitVec 32)) (y (_ BitVec 32))) (_ BitVec 32)
// CHECK-NEXT:   (bvor x y))
