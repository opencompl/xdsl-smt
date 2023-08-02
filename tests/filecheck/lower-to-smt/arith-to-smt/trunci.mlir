// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt | filecheck "%s"

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32):
    %r = "arith.trunci"(%x) : (i32) -> i16
    "func.return"(%r) : (i16) -> ()
  }) {"sym_name" = "test", "function_type" = (i32) -> i16, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      (define-fun tmp ((x (_ BitVec 32))) (_ BitVec 16)
// CHECK-NEXT:   ((_ extract 15 0) x))
