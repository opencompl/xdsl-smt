// XFAIL: *
// RUN: xdsl-smt "%s" -p=lower-to-smt,canonicalize,dce -t=smt | filecheck "%s"


builtin.module {
  func.func private @constant() -> i32 {
    %x = "llvm.mlir.constant"() <{value = 3 : i32}> : () -> i32
    func.return %x : i32
  }
}

// CHECK:      (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT: (define-fun constant () (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:   (pair (_ bv3 32) false))
