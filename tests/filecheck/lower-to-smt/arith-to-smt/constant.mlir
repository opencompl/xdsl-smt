// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize-smt -t=smt | filecheck "%s"


"builtin.module"() ({
  "func.func"() ({
    %x = "arith.constant"() {"value" = 3 : i32} : () -> i32
    "func.return"(%x) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = () -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()


// CHECK:      (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT: (define-fun tmp () (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:   (pair (_ bv3 32) false))
