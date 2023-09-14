// RUN: xdsl-smt "%s" -p=dce | filecheck "%s"

"builtin.module"() ({

  %three = "smt.bv.constant"() {"value" = #smt.bv.bv_val<3: 32>} : () -> !smt.bv.bv<32>
  // CHECK:   %three = "smt.bv.constant"() {"value" = #smt.bv.bv_val<3: 32>} : () -> !smt.bv.bv<32>

  %eq = "smt.eq"(%three, %three) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  // CHECK-NEXT: %eq = "smt.eq"(%three, %three) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool

  // UNUSED
  %four = "smt.bv.constant"() {"value" = #smt.bv.bv_val<4: 32>} : () -> !smt.bv.bv<32>

  "smt.assert"(%eq) : (!smt.bool) -> ()
  // CHECK-NEXT: "smt.assert"(%eq) : (!smt.bool) -> ()
}) : () -> ()
