// RUN: xdsl-smt "%s" -p=lower-to-smt -t=smt | filecheck "%s"

"builtin.module"() ({
  "llvm.func"() <{CConv = #llvm.cconv<ccc>, function_type = !llvm.func<i32 (i32)>, linkage = #llvm.linkage<external>, sym_name = "test", visibility_ = 0 : i64}> ({
  ^bb0(%arg0: i32):
    "llvm.return"(%arg0) : (i32) -> ()
  }) : () -> ()
}) {llvm.data_layout = ""} : () -> ()


// CHECK:      (declare-datatypes ((Pair 2)) ((par (X Y) ((pair (first X) (second Y))))))
// CHECK-NEXT: (define-fun test ((arg0 (Pair (_ BitVec 32) Bool))) (Pair (_ BitVec 32) Bool)
// CHECK-NEXT:   arg0)
