// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,smt-expand,canonicalize,dce,merge-func-results -t=smt | z3 -in

builtin.module {
  func.func private @main(%seed : !transfer.integer) -> !transfer.integer {
    %x = "transfer.constant"(%seed) {value = 8 : index} : (!transfer.integer) -> !transfer.integer
    %r = "transfer.countl_zero"(%x) : (!transfer.integer) -> !transfer.integer
    func.return %r : !transfer.integer
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%seed : !smt.bv<8>, %1 : !effect.state):
// CHECK-NEXT:      %x = smt.bv.constant #smt.bv<8> : !smt.bv<8>
// CHECK-NEXT:      %2 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
// CHECK-NEXT:      %3 = smt.bv.constant #smt.bv<4> : !smt.bv<8>
// CHECK-NEXT:      %4 = smt.bv.constant #smt.bv<4> : !smt.bv<8>
// CHECK-NEXT:      %5 = "smt.bv.shl"(%x, %3) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %6 = smt.bv.constant #smt.bv<192> : !smt.bv<8>
// CHECK-NEXT:      %7 = smt.bv.constant #smt.bv<2> : !smt.bv<8>
// CHECK-NEXT:      %8 = "smt.bv.and"(%5, %6) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %9 = "smt.eq"(%8, %2) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %10 = smt.bv.constant #smt.bv<6> : !smt.bv<8>
// CHECK-NEXT:      %11 = "smt.bv.shl"(%5, %7) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %12 = "smt.ite"(%9, %10, %4) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %13 = "smt.ite"(%9, %11, %5) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %14 = smt.bv.constant #smt.bv<128> : !smt.bv<8>
// CHECK-NEXT:      %15 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:      %16 = "smt.bv.and"(%13, %14) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %17 = "smt.eq"(%16, %2) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %18 = "smt.bv.add"(%12, %15) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %r = "smt.ite"(%17, %18, %12) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      "smt.return"(%r, %1) : (!smt.bv<8>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "main"} : () -> ((!smt.bv<8>, !effect.state) -> (!smt.bv<8>, !effect.state))
// CHECK-NEXT:  }
