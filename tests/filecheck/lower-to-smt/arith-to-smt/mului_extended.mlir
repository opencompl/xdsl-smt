// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize-smt | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,canonicalize-smt -t=smt | z3 -in

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : i32, %y : i32):
    %low, %high = arith.mului_extended %x, %y : i32
    func.return %low, %high : i32, i32
  }) {"sym_name" = "test", "function_type" = (i32, i32) -> (i32, i32), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      (define-fun test ((x (Pair (_ BitVec 32) Bool)) (y (Pair (_ BitVec 32) Bool))) (Pair (Pair (_ BitVec 32) Bool) (Pair (_ BitVec 32) Bool))
// CHECK-NEXT:   (let ((tmp (or (second x) (second y))))
// CHECK-NEXT:   (let ((tmp_0 (bvmul ((_ zero_extend 32) (first x)) ((_ zero_extend 32) (first y)))))
// CHECK-NEXT:   (pair (pair ((_ extract 31 0) tmp_0) tmp) (pair ((_ extract 63 32) tmp_0) tmp)))))
