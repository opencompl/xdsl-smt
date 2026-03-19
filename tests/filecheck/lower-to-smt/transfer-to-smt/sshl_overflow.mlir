// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,smt-expand,canonicalize,dce,merge-func-results -t=smt

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.sshl_overflow"(%x, %y) : (!transfer.integer,!transfer.integer) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer, !transfer.integer) -> i1, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       %3 = "smt.bv.uge"(%y, %2) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:  %4 = "smt.bv.shl"(%x, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:  %5 = "smt.bv.ashr"(%4, %y) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:  %6 = "smt.distinct"(%5, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:  %7 = smt.or %3, %6
// CHECK-NEXT:  %8 = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK-NEXT:  %9 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:  %10 = "smt.ite"(%7, %8, %9) : (!smt.bool, !smt.bv<1>, !smt.bv<1>) -> !smt.bv<1>
// CHECK-NEXT:  %11 = smt.constant false
// CHECK-NEXT:  %r = "smt.utils.pair"(%10, %11) : (!smt.bv<1>, !smt.bool) -> !smt.utils.pair<!smt.bv<1>, !smt.bool>
// CHECK-NEXT:  "smt.return"(%r, %1) : (!smt.utils.pair<!smt.bv<1>, !smt.bool>, !effect.state) -> ()
