// RUN: xdsl-smt "%s" | xdsl-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt "%s" -t=smt | z3 -in

"builtin.module"() ({

  %rne_full = "smt.fp.round_nearest_ties_to_even"() : () -> !smt.fp.rounding_mode
  %three = smt.bv.constant #smt.bv<3> : !smt.bv<8>
  %four = smt.bv.constant #smt.bv<4> : !smt.bv<10>
  %zero = smt.bv.constant #smt.bv<0> : !smt.bv<1>
  %random_fp = "smt.fp.constant"(%zero, %three, %four) : (!smt.bv<1>, !smt.bv<8>, !smt.bv<10>) -> !smt.fp<8,11>
  %pzero = "smt.fp.pzero"() : () -> !smt.fp<8,11>

  %abs = "smt.fp.abs"(%random_fp): (!smt.fp<8, 11>) -> !smt.fp<8,11>
  %neg = "smt.fp.neg"(%random_fp): (!smt.fp<8, 11>) -> !smt.fp<8,11>
  %sqrt = "smt.fp.sqrt"(%rne_full, %random_fp): (!smt.fp.rounding_mode, !smt.fp<8, 11>) -> !smt.fp<8,11>
  %roundToIntegral = "smt.fp.roundToIntegral"(%rne_full, %random_fp): (!smt.fp.rounding_mode, !smt.fp<8, 11>) -> !smt.fp<8,11>

  %eq_abs_neg= "smt.eq"(%abs, %neg) : (!smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.bool
  %eq_sqrt_roundToIntegral= "smt.eq"(%sqrt, %roundToIntegral) : (!smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.bool

  "smt.assert"(%eq_abs_neg) : (!smt.bool) -> ()
  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK: (= (fp.abs $random_fp) (fp.neg $random_fp))))
  "smt.assert"(%eq_sqrt_roundToIntegral) : (!smt.bool) -> ()
  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK: (let (($rne_full roundNearestTiesToEven))
  // CHECK: (= (fp.sqrt $rne_full $random_fp) (fp.roundToIntegral $rne_full $random_fp)))))
}) : () -> ()
