// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize-smt -t=smt | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    // TODO: return the overflow bit once we can lower functions with multiple return values
    %r, %o = "arith.addui_carry"(%x, %y) : (i32, i32) -> (i32, i1)
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      (define-fun tmp ((x (_ BitVec 32)) (y (_ BitVec 32))) (_ BitVec 32)
// CHECK-NEXT:   ((_ extract 31 0) (bvadd (concat (_ bv0 1) x) (concat (_ bv0 1) y))))
