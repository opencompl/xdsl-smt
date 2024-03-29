// RUN: xdsl-smt "%s" -p dce -o "%t" -t mlir && xdsl-tv "%s" "%t" | filecheck "%s"
// RUN: xdsl-smt "%s" -p dce -o "%t" -t mlir && xdsl-tv "%s" "%t" | z3 -in

"builtin.module"() ({
  "func.func"() ({
    %x = "arith.constant"() {"value" = 3 : i32} : () -> i32
    %unused = "arith.constant"() {"value" = 42 : i32} : () -> i32
    "func.return"(%x) : (i32) -> ()
  }) {"sym_name" = "test", "function_type" = () -> i32, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT:  (define-fun test () (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (pair (_ bv3 32) false))
// CHECK-NEXT:  (define-fun test_0 () (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:    (pair (_ bv3 32) false))
// CHECK-NEXT:  (assert (let ((tmp test))
// CHECK-NEXT:    (let ((tmp_0 test_0))
// CHECK-NEXT:    (not (or (and (not (second tmp_0)) (= (first tmp) (first tmp_0))) (second tmp))))))
// CHECK-NEXT:  (check-sat)
