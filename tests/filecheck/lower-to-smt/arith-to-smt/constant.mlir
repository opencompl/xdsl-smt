// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt | filecheck "%s"


"builtin.module"() ({
  "func.func"() ({
    %x = "arith.constant"() {"value" = 3 : i32} : () -> i32
    "func.return"(%x) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = () -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()


// CHECK:      (define-fun tmp () (_ BitVec 32)
// CHECK-NEXT:   (_ bv3 32))
