// RUN: xdsl-smt %s -p=resolve-transfer-widths{width=8},lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=resolve-transfer-widths{width=8},lower-to-smt,lower-effects,smt-expand,canonicalize,dce,merge-func-results -t=smt

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.ushl_overflow"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer<1>
    "func.return"(%r) : (!transfer.integer<1>) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer<1>, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.bv<8>, %y : !smt.bv<8>, %1 : !effect.state):
// CHECK-NEXT:      %2 = smt.bv.constant #smt.bv<8> : !smt.bv<8>
// CHECK-NEXT:      %3 = "smt.bv.uge"(%y, %2) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %4 = "smt.declare_const"() : () -> !smt.bv<8>
// CHECK-NEXT:      %5 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
// CHECK-NEXT:      %6 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:      %7 = "smt.eq"(%5, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %8 = smt.bv.constant #smt.bv<255> : !smt.bv<8>
// CHECK-NEXT:      %9 = "smt.eq"(%8, %4) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %10 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:      %11 = smt.bv.constant #smt.bv<2> : !smt.bv<8>
// CHECK-NEXT:      %12 = smt.bv.constant #smt.bv<4> : !smt.bv<8>
// CHECK-NEXT:      %13 = "smt.bv.ashr"(%x, %10) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %14 = "smt.bv.ashr"(%x, %11) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %15 = "smt.bv.ashr"(%x, %12) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %16 = "smt.bv.or"(%x, %13) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %17 = "smt.bv.or"(%16, %14) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %18 = "smt.bv.or"(%17, %15) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %19 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:      %20 = "smt.bv.lshr"(%18, %19) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %21 = "smt.bv.sub"(%18, %20) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %22 = "smt.bv.shl"(%6, %4) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %23 = "smt.eq"(%21, %22) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %24 = "smt.ite"(%7, %9, %23) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      "smt.assert"(%24) : (!smt.bool) -> ()
// CHECK-NEXT:      %25 = smt.bv.constant #smt.bv<7> : !smt.bv<8>
// CHECK-NEXT:      %26 = "smt.bv.sub"(%25, %4) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %27 = "smt.bv.ugt"(%y, %26) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %28 = smt.or %3, %27
// CHECK-NEXT:      %29 = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK-NEXT:      %30 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:      %r = "smt.ite"(%28, %29, %30) : (!smt.bool, !smt.bv<1>, !smt.bv<1>) -> !smt.bv<1>
// CHECK-NEXT:      "smt.return"(%r, %1) : (!smt.bv<1>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "test"} : () -> ((!smt.bv<8>, !smt.bv<8>, !effect.state) -> (!smt.bv<1>, !effect.state))
// CHECK-NEXT:  }
