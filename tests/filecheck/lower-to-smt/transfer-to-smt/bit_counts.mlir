// RUN: xdsl-smt %s -p=lower-to-smt,canonicalize,dce | filecheck %s
// RUN: xdsl-smt %s -p=lower-to-smt,lower-effects,smt-expand,canonicalize,dce,merge-func-results -t=smt | z3 -in

builtin.module {
  func.func @main(%x : !transfer.integer) -> (!transfer.integer, !transfer.integer, !transfer.integer, !transfer.integer) {
    %0 = "transfer.countl_zero"(%x) : (!transfer.integer) -> !transfer.integer
    %1 = "transfer.countr_zero"(%x) : (!transfer.integer) -> !transfer.integer
    %2 = "transfer.countl_one"(%x) : (!transfer.integer) -> !transfer.integer
    %3 = "transfer.countr_one"(%x) : (!transfer.integer) -> !transfer.integer
    func.return %0, %1, %2, %3 : !transfer.integer, !transfer.integer, !transfer.integer, !transfer.integer
  }
}

// CHECK:       builtin.module {
// CHECK-NEXT:    %0 = "smt.define_fun"() ({
// CHECK-NEXT:    ^0(%x : !smt.bv<8>, %1 : !effect.state):
// CHECK-NEXT:      %2 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
// CHECK-NEXT:      %3 = smt.bv.constant #smt.bv<8> : !smt.bv<8>
// CHECK-NEXT:      %4 = smt.bv.constant #smt.bv<240> : !smt.bv<8>
// CHECK-NEXT:      %5 = smt.bv.constant #smt.bv<4> : !smt.bv<8>
// CHECK-NEXT:      %6 = "smt.bv.and"(%x, %4) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %7 = "smt.eq"(%6, %2) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %8 = smt.bv.constant #smt.bv<4> : !smt.bv<8>
// CHECK-NEXT:      %9 = "smt.bv.shl"(%x, %5) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %10 = "smt.ite"(%7, %8, %2) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %11 = "smt.ite"(%7, %9, %x) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %12 = smt.bv.constant #smt.bv<192> : !smt.bv<8>
// CHECK-NEXT:      %13 = smt.bv.constant #smt.bv<2> : !smt.bv<8>
// CHECK-NEXT:      %14 = "smt.bv.and"(%11, %12) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %15 = "smt.eq"(%14, %2) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %16 = "smt.bv.add"(%10, %13) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %17 = "smt.bv.shl"(%11, %13) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %18 = "smt.ite"(%15, %16, %10) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %19 = "smt.ite"(%15, %17, %11) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %20 = smt.bv.constant #smt.bv<128> : !smt.bv<8>
// CHECK-NEXT:      %21 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:      %22 = "smt.bv.and"(%19, %20) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %23 = "smt.eq"(%22, %2) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %24 = "smt.bv.add"(%18, %21) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %25 = "smt.ite"(%23, %24, %18) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %26 = "smt.eq"(%x, %2) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %27 = "smt.ite"(%26, %3, %25) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %28 = "smt.ite"(%29, %30, %31) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %31 = "smt.bv.sub"(%30, %32) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %29 = "smt.eq"(%33, %34) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %33 = "smt.bv.and"(%35, %36) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %32 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:      %36 = smt.bv.constant #smt.bv<85> : !smt.bv<8>
// CHECK-NEXT:      %30 = "smt.ite"(%37, %38, %39) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %39 = "smt.bv.sub"(%38, %40) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %37 = "smt.eq"(%41, %34) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %41 = "smt.bv.and"(%35, %42) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %40 = smt.bv.constant #smt.bv<2> : !smt.bv<8>
// CHECK-NEXT:      %42 = smt.bv.constant #smt.bv<51> : !smt.bv<8>
// CHECK-NEXT:      %38 = "smt.ite"(%43, %44, %45) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %45 = "smt.bv.sub"(%44, %46) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %43 = "smt.eq"(%47, %34) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %47 = "smt.bv.and"(%35, %48) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %46 = smt.bv.constant #smt.bv<4> : !smt.bv<8>
// CHECK-NEXT:      %48 = smt.bv.constant #smt.bv<15> : !smt.bv<8>
// CHECK-NEXT:      %44 = "smt.ite"(%49, %50, %51) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %49 = "smt.eq"(%x, %34) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %35 = "smt.bv.and"(%x, %52) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %52 = "smt.bv.sub"(%34, %x) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %51 = smt.bv.constant #smt.bv<7> : !smt.bv<8>
// CHECK-NEXT:      %50 = smt.bv.constant #smt.bv<8> : !smt.bv<8>
// CHECK-NEXT:      %34 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
// CHECK-NEXT:      %53 = "smt.bv.not"(%x) : (!smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %54 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
// CHECK-NEXT:      %55 = smt.bv.constant #smt.bv<8> : !smt.bv<8>
// CHECK-NEXT:      %56 = smt.bv.constant #smt.bv<240> : !smt.bv<8>
// CHECK-NEXT:      %57 = smt.bv.constant #smt.bv<4> : !smt.bv<8>
// CHECK-NEXT:      %58 = "smt.bv.and"(%53, %56) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %59 = "smt.eq"(%58, %54) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %60 = smt.bv.constant #smt.bv<4> : !smt.bv<8>
// CHECK-NEXT:      %61 = "smt.bv.shl"(%53, %57) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %62 = "smt.ite"(%59, %60, %54) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %63 = "smt.ite"(%59, %61, %53) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %64 = smt.bv.constant #smt.bv<192> : !smt.bv<8>
// CHECK-NEXT:      %65 = smt.bv.constant #smt.bv<2> : !smt.bv<8>
// CHECK-NEXT:      %66 = "smt.bv.and"(%63, %64) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %67 = "smt.eq"(%66, %54) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %68 = "smt.bv.add"(%62, %65) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %69 = "smt.bv.shl"(%63, %65) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %70 = "smt.ite"(%67, %68, %62) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %71 = "smt.ite"(%67, %69, %63) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %72 = smt.bv.constant #smt.bv<128> : !smt.bv<8>
// CHECK-NEXT:      %73 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:      %74 = "smt.bv.and"(%71, %72) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %75 = "smt.eq"(%74, %54) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %76 = "smt.bv.add"(%70, %73) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %77 = "smt.ite"(%75, %76, %70) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %78 = "smt.eq"(%53, %54) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %79 = "smt.ite"(%78, %55, %77) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %80 = "smt.bv.not"(%x) : (!smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %81 = smt.bv.constant #smt.bv<0> : !smt.bv<8>
// CHECK-NEXT:      %82 = smt.bv.constant #smt.bv<8> : !smt.bv<8>
// CHECK-NEXT:      %83 = smt.bv.constant #smt.bv<7> : !smt.bv<8>
// CHECK-NEXT:      %84 = "smt.bv.sub"(%81, %80) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %85 = "smt.bv.and"(%80, %84) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %86 = "smt.eq"(%80, %81) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %87 = "smt.ite"(%86, %82, %83) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %88 = smt.bv.constant #smt.bv<15> : !smt.bv<8>
// CHECK-NEXT:      %89 = smt.bv.constant #smt.bv<4> : !smt.bv<8>
// CHECK-NEXT:      %90 = "smt.bv.and"(%85, %88) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %91 = "smt.eq"(%90, %81) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %92 = "smt.bv.sub"(%87, %89) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %93 = "smt.ite"(%91, %87, %92) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %94 = smt.bv.constant #smt.bv<51> : !smt.bv<8>
// CHECK-NEXT:      %95 = smt.bv.constant #smt.bv<2> : !smt.bv<8>
// CHECK-NEXT:      %96 = "smt.bv.and"(%85, %94) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %97 = "smt.eq"(%96, %81) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %98 = "smt.bv.sub"(%93, %95) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %99 = "smt.ite"(%97, %93, %98) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %100 = smt.bv.constant #smt.bv<85> : !smt.bv<8>
// CHECK-NEXT:      %101 = smt.bv.constant #smt.bv<1> : !smt.bv<8>
// CHECK-NEXT:      %102 = "smt.bv.and"(%85, %100) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %103 = "smt.eq"(%102, %81) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bool
// CHECK-NEXT:      %104 = "smt.bv.sub"(%99, %101) : (!smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      %105 = "smt.ite"(%103, %99, %104) : (!smt.bool, !smt.bv<8>, !smt.bv<8>) -> !smt.bv<8>
// CHECK-NEXT:      "smt.return"(%27, %28, %79, %105, %1) : (!smt.bv<8>, !smt.bv<8>, !smt.bv<8>, !smt.bv<8>, !effect.state) -> ()
// CHECK-NEXT:    }) {fun_name = "main"} : () -> ((!smt.bv<8>, !effect.state) -> (!smt.bv<8>, !smt.bv<8>, !smt.bv<8>, !smt.bv<8>, !effect.state))
// CHECK-NEXT:  }
