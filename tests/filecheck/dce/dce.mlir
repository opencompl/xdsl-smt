// RUN: xdsl-smt "%s" -p=dce | filecheck "%s"

"builtin.module"() ({

  %three = smt.bv.constant #smt.bv<3> : !smt.bv<32>
  // CHECK:   %three = smt.bv.constant #smt.bv<3> : !smt.bv<32>

  %eq = "smt.eq"(%three, %three) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool
  // CHECK-NEXT: %eq = "smt.eq"(%three, %three) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bool

  // UNUSED
  %four = smt.bv.constant #smt.bv<4> : !smt.bv<32>

  "smt.assert"(%eq) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%eq) : (!smt.bool) -> ()
}) : () -> ()
