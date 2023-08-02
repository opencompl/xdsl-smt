// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt | filecheck "%s"

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i16):
    %r = "arith.extsi"(%x) : (i16) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = (i16) -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      (define-fun tmp ((x (_ BitVec 16))) (_ BitVec 32)
// CHECK-NEXT:   (concat (ite (= ((_ extract 15 15) x) (_ bv0 1)) (_ bv0 16) (_ bv65535 16)) x))
