// RUN: xdsl-smt %s -p=resolve-transfer-widths{width-map=\"default=8\"},lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=resolve-transfer-widths{width-map=\"default=8\"},lower-to-smt,lower-effects,smt-expand,canonicalize,dce,merge-func-results -t=smt

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer<@W>, %y : !transfer.integer<@W>):
    %r = "transfer.sshl_overflow"(%x, %y) : (!transfer.integer<@W>,!transfer.integer<@W>) -> i1
    "func.return"(%r) : (i1) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer<@W>, !transfer.integer<@W>) -> i1, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       %3 = "smt.bv.uge"(%y, %2) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK:       %5 = "smt.bv.sge"(%x, %4) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK:       %29 = "smt.bv.uge"(%y, %28) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK:       %31 = "smt.bv.slt"(%x, %30) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK:       %56 = "smt.bv.uge"(%y, %55) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:    %57 = smt.and %5, %29
// CHECK-NEXT:    %58 = smt.and %31, %56
// CHECK-NEXT:    %59 = smt.or %57, %58
// CHECK-NEXT:    %60 = smt.or %3, %59
// CHECK-NEXT:    %61 = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK-NEXT:    %62 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:    %63 = "smt.ite"(%60, %61, %62) : (!smt.bool, !smt.bv<1>, !smt.bv<1>) -> !smt.bv<1>
// CHECK-NEXT:    %64 = smt.constant false
// CHECK-NEXT:    %r = "smt.utils.pair"(%63, %64) : (!smt.bv<1>, !smt.bool) -> !smt.utils.pair<!smt.bv<1>, !smt.bool>
// CHECK-NEXT:    smt.return"(%r, %1) : (!smt.utils.pair<!smt.bv<1>, !smt.bool>, !effect.state) -> ()
