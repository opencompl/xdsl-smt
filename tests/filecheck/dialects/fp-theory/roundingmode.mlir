// RUN: xdsl-smt "%s" | xdsl-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt "%s" -t=smt | z3 -in

"builtin.module"() ({

  %rne_full = "smt.fp.round_nearest_ties_to_even"() : () -> !smt.fp.rounding_mode
  %rne = "smt.fp.rne"() : () -> !smt.fp.rounding_mode
  %rna_full = "smt.fp.round_nearest_ties_to_away"() : () -> !smt.fp.rounding_mode
  %rna = "smt.fp.rna"() : () -> !smt.fp.rounding_mode
  %rtp_full = "smt.fp.round_toward_positive"() : () -> !smt.fp.rounding_mode
  %rtp = "smt.fp.rtp"() : () -> !smt.fp.rounding_mode
  %rtn_full = "smt.fp.round_toward_negative"() : () -> !smt.fp.rounding_mode
  %rtn = "smt.fp.rtn"() : () -> !smt.fp.rounding_mode
  %rtz_full = "smt.fp.round_toward_zero"() : () -> !smt.fp.rounding_mode
  %rtz = "smt.fp.rtz"() : () -> !smt.fp.rounding_mode


  %eq_rne = "smt.eq"(%rne_full, %rne) : (!smt.fp.rounding_mode, !smt.fp.rounding_mode) -> !smt.bool
  %eq_rna = "smt.eq"(%rna_full, %rna) : (!smt.fp.rounding_mode, !smt.fp.rounding_mode) -> !smt.bool
  %eq_rtp = "smt.eq"(%rtp_full, %rtp) : (!smt.fp.rounding_mode, !smt.fp.rounding_mode) -> !smt.bool
  %eq_rtn = "smt.eq"(%rtn_full, %rtn) : (!smt.fp.rounding_mode, !smt.fp.rounding_mode) -> !smt.bool
  %eq_rtz = "smt.eq"(%rtz_full, %rtz) : (!smt.fp.rounding_mode, !smt.fp.rounding_mode) -> !smt.bool


  "smt.assert"(%eq_rne) : (!smt.bool) -> ()
  // CHECK: (assert (= roundNearestTiesToEven RNE))
  "smt.assert"(%eq_rna) : (!smt.bool) -> ()
  // CHECK: (assert (= roundNearestTiesToAway RNA))
  "smt.assert"(%eq_rtp) : (!smt.bool) -> ()
  // CHECK: (assert (= roundTowardPositive RTP))
  "smt.assert"(%eq_rtn) : (!smt.bool) -> ()
  // CHECK: (assert (= roundTowardNegative RTN))
  "smt.assert"(%eq_rtz) : (!smt.bool) -> ()
  // CHECK: (assert (= roundTowardZero RTZ))
}) : () -> ()
