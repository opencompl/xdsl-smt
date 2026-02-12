// RUN: xdsl-smt %s -p=resolve-transfer-widths{width=8},lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=resolve-transfer-widths{width=8},lower-to-smt,lower-effects,smt-expand,canonicalize,dce,merge-func-results -t=smt

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.sshl_overflow"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer<1>
    "func.return"(%r) : (!transfer.integer<1>) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer<1>, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.bv<8>, %y : !smt.bv<8>, %1 : !effect.state):
// CHECK-NEXT:      %2 = smt.bv.constant #smt.bv<8> : !smt.bv<8>
// CHECK-NEXT:      %3 = "smt.bv.uge"(%y, %2) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %4 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
// CHECK-NEXT:      %5 = "smt.bv.sge"(%x, %4) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %6 = "smt.declare_const"() : () -> !smt.bv<8>
// CHECK-NEXT:      %7 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
// CHECK-NEXT:      %8 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:      %9 = "smt.eq"(%7, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %10 = smt.bv.constant #smt.bv<255> : !smt.bv<8>
// CHECK-NEXT:      %11 = "smt.eq"(%10, %6) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %12 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:      %13 = smt.bv.constant #smt.bv<2> : !smt.bv<8>
// CHECK-NEXT:      %14 = smt.bv.constant #smt.bv<4> : !smt.bv<8>
// CHECK-NEXT:      %15 = "smt.bv.ashr"(%x, %12) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %16 = "smt.bv.ashr"(%x, %13) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %17 = "smt.bv.ashr"(%x, %14) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %18 = "smt.bv.or"(%x, %15) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %19 = "smt.bv.or"(%18, %16) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %20 = "smt.bv.or"(%19, %17) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %21 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:      %22 = "smt.bv.lshr"(%20, %21) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %23 = "smt.bv.sub"(%20, %22) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %24 = "smt.bv.shl"(%8, %6) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %25 = "smt.eq"(%23, %24) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %26 = "smt.ite"(%9, %11, %25) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      "smt.assert"(%26) : (!smt.bool) -> ()
// CHECK-NEXT:      %27 = smt.bv.constant #smt.bv<7> : !smt.bv<8>
// CHECK-NEXT:      %28 = "smt.bv.sub"(%27, %6) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %29 = "smt.bv.uge"(%y, %28) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %30 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
// CHECK-NEXT:      %31 = "smt.bv.slt"(%x, %30) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %32 = "smt.bv.not"(%x) : (!smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %33 = "smt.declare_const"() : () -> !smt.bv<8>
// CHECK-NEXT:      %34 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
// CHECK-NEXT:      %35 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:      %36 = "smt.eq"(%34, %32) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %37 = smt.bv.constant #smt.bv<255> : !smt.bv<8>
// CHECK-NEXT:      %38 = "smt.eq"(%37, %33) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %39 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:      %40 = smt.bv.constant #smt.bv<2> : !smt.bv<8>
// CHECK-NEXT:      %41 = smt.bv.constant #smt.bv<4> : !smt.bv<8>
// CHECK-NEXT:      %42 = "smt.bv.ashr"(%32, %39) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %43 = "smt.bv.ashr"(%32, %40) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %44 = "smt.bv.ashr"(%32, %41) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %45 = "smt.bv.or"(%32, %42) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %46 = "smt.bv.or"(%45, %43) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %47 = "smt.bv.or"(%46, %44) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %48 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:      %49 = "smt.bv.lshr"(%47, %48) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %50 = "smt.bv.sub"(%47, %49) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %51 = "smt.bv.shl"(%35, %33) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %52 = "smt.eq"(%50, %51) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %53 = "smt.ite"(%36, %38, %52) : (!smt.bool, !smt.bool, !smt.bool) -> !smt.bool
// CHECK-NEXT:      "smt.assert"(%53) : (!smt.bool) -> ()
// CHECK-NEXT:      %54 = smt.bv.constant #smt.bv<7> : !smt.bv<8>
// CHECK-NEXT:      %55 = "smt.bv.sub"(%54, %33) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %56 = "smt.bv.uge"(%y, %55) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %57 = smt.and %5, %29
// CHECK-NEXT:      %58 = smt.and %31, %56
// CHECK-NEXT:      %59 = smt.or %57, %58
// CHECK-NEXT:      %60 = smt.or %3, %59
// CHECK-NEXT:      %61 = smt.bv.constant #smt.bv<1> : !smt.bv<1>
// CHECK-NEXT:      %62 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:      %r = "smt.ite"(%60, %61, %62) : (!smt.bool, !smt.bv<1>, !smt.bv<1>) -> !smt.bv<1>
// CHECK-NEXT:      "smt.return"(%r, %1) : (!smt.bv<1>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "test"} : () -> ((!smt.bv<8>, !smt.bv<8>, !effect.state) -> (!smt.bv<1>, !effect.state))
// CHECK-NEXT:  }
