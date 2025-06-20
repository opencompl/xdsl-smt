// RUN: xdsl-smt "%s" | xdsl-smt -t mlir --print-op-generic | filecheck "%s"

"builtin.module"() ({

  "func.func"() ({
  ^0(%x : i32, %y : i32):
    // requires: not x = y
    "hoare.requires"() ({
    ^1(%x2 : !smt.bv<32>, %y2 : !smt.bv<32>):
      %x2neg = "smt.bv.neg"(%x2) : (!smt.bv<32>) -> !smt.bv<32>
      %eq = "smt.eq"(%x2neg, %y2) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
      "hoare.yield"(%eq) : (!smt.bool) -> ()
    }) : () -> ()

    // ensures: res = 0
    "hoare.ensures"() ({
    ^2(%x3 : !smt.bv<32>, %y3 : !smt.bv<32>, %r3 : !smt.bv<32>):
      %zero = "smt.bv.constant"() {value = #smt.bv<0> : !smt.bv<32>} : () -> !smt.bv<32>
      %eq_1 = "smt.eq"(%r3, %zero) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
      "hoare.yield"(%eq_1) : (!smt.bool) -> ()
    }) : () -> ()

    // r = x + y
    %r = "arith.addi"(%x, %y) : (i32, i32) -> i32
    "func.return"(%r) : (i32) -> ()
  }) {sym_name = "test", function_type = (i32, i32) -> i32, sym_visibility = "private"} : () -> ()
}) : () -> ()



// CHECK:      "builtin.module"() ({
// CHECK-NEXT:   "func.func"() <{sym_name = "test", function_type = (i32, i32) -> i32, sym_visibility = "private"}> ({
// CHECK-NEXT:   ^0(%x : i32, %y : i32):
// CHECK-NEXT:     "hoare.requires"() ({
// CHECK-NEXT:     ^1(%x2 : !smt.bv<32>, %y2 : !smt.bv<32>):
// CHECK-NEXT:       %x2neg = "smt.bv.neg"(%x2) : (!smt.bv<32>) -> !smt.bv<32>
// CHECK-NEXT:       %eq = "smt.eq"(%x2neg, %y2) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:       "hoare.yield"(%eq) : (!smt.bool) -> ()
// CHECK-NEXT:     }) : () -> ()
// CHECK-NEXT:     "hoare.ensures"() ({
// CHECK-NEXT:     ^2(%x3 : !smt.bv<32>, %y3 : !smt.bv<32>, %r3 : !smt.bv<32>):
// CHECK-NEXT:       %zero = "smt.bv.constant"() {value = #smt.bv<0> : !smt.bv<32>} : () -> !smt.bv<32>
// CHECK-NEXT:       %eq_1 = "smt.eq"(%r3, %zero) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
// CHECK-NEXT:       "hoare.yield"(%eq_1) : (!smt.bool) -> ()
// CHECK-NEXT:     }) : () -> ()
// CHECK-NEXT:     %r = "arith.addi"(%x, %y) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
// CHECK-NEXT:     "func.return"(%r) : (i32) -> ()
// CHECK-NEXT:   })
// CHECK-NEXT: }) : () -> ()
