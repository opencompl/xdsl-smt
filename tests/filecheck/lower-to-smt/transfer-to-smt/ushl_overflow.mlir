// RUN: xdsl-smt %s -p=resolve-transfer-widths,lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=resolve-transfer-widths,lower-to-smt,lower-effects,smt-expand,canonicalize,dce,merge-func-results -t=smt

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer<@W>, %y : !transfer.integer<@W>):
    %r = "transfer.ushl_overflow"(%x, %y) : (!transfer.integer<@W>,!transfer.integer<@W>) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer<@W>, !transfer.integer<@W>) -> i1, "sym_visibility" = "private"} : () -> ()
}) {"transfer.widths" = {W = 8 : index}} : () -> ()

// CHECK:       %3 = "smt.bv.uge"(%y, %2) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK:       %27 = "smt.bv.ugt"(%y, %26) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:  %28 = smt.or %3, %27
// CHECK-NEXT:  %29 = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK-NEXT:  %30 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:  %31 = "smt.ite"(%28, %29, %30) : (!smt.bool, !smt.bv<1>, !smt.bv<1>) -> !smt.bv<1>
// CHECK-NEXT:  smt.constant false
// CHECK-NEXT:  %r = "smt.utils.pair"(%31, %32) : (!smt.bv<1>, !smt.bool) -> !smt.utils.pair<!smt.bv<1>, !smt.bool>
// CHECK-NEXT:  "smt.return"(%r, %1) : (!smt.utils.pair<!smt.bv<1>, !smt.bool>, !effect.state) -> ()
