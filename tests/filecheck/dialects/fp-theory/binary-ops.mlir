// RUN: xdsl-smt "%s" | xdsl-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt "%s" -t=smt | z3 -in

"builtin.module"() ({

  %rne_full = "smt.fp.round_nearest_ties_to_even"() : () -> !smt.fp.rounding_mode
  %three = smt.bv.constant #smt.bv<3> : !smt.bv<8>
  %four = smt.bv.constant #smt.bv<4> : !smt.bv<10>
  %zero = smt.bv.constant #smt.bv<0> : !smt.bv<1>
  %random_fp = "smt.fp.constant"(%zero, %three, %four) : (!smt.bv<1>, !smt.bv<8>, !smt.bv<10>) -> !smt.fp<8,11>
  %pzero = "smt.fp.pzero"() : () -> !smt.fp<8,11>

  %max = "smt.fp.max"(%random_fp, %pzero): (!smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.fp<8,11>
  %min = "smt.fp.max"(%random_fp, %pzero): (!smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.fp<8,11>
  %rem = "smt.fp.max"(%random_fp, %random_fp): (!smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.fp<8,11>

  %add = "smt.fp.add"(%rne_full, %random_fp, %pzero): (!smt.fp.rounding_mode, !smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.fp<8,11>
  %sub = "smt.fp.sub"(%rne_full, %random_fp, %pzero): (!smt.fp.rounding_mode, !smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.fp<8,11>
  %mul = "smt.fp.mul"(%rne_full, %random_fp, %pzero): (!smt.fp.rounding_mode, !smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.fp<8,11>
  %div = "smt.fp.div"(%rne_full, %random_fp, %pzero): (!smt.fp.rounding_mode, !smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.fp<8,11>

  %eq_max_min= "smt.eq"(%max, %min) : (!smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.bool
  %eq_rem_rem= "smt.eq"(%rem, %rem) : (!smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.bool
  %eq_add_sub= "smt.eq"(%add, %sub) : (!smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.bool
  %eq_mul_div= "smt.eq"(%mul, %div) : (!smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.bool

  "smt.assert"(%eq_max_min) : (!smt.bool) -> ()
  // CHECK: (assert (let (($pzero (_ +zero 8 11)))
  // CHECK: (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK: (= (fp.max $random_fp $pzero) (fp.max $random_fp $pzero)))))
  "smt.assert"(%eq_rem_rem) : (!smt.bool) -> ()
  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK: (let (($rem (fp.max $random_fp $random_fp)))
  // CHECK: (= $rem $rem))))
  "smt.assert"(%eq_add_sub) : (!smt.bool) -> ()
  // CHECK: (assert (let (($pzero (_ +zero 8 11)))
  // CHECK: (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK: (let (($rne_full roundNearestTiesToEven))
  // CHECK: (= (fp.add $rne_full $random_fp $pzero) (fp.sub $rne_full $random_fp $pzero))))))
  "smt.assert"(%eq_mul_div) : (!smt.bool) -> ()
  // CHECK: (assert (let (($pzero (_ +zero 8 11)))
  // CHECK: (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK: (let (($rne_full roundNearestTiesToEven))
  // CHECK: (= (fp.mul $rne_full $random_fp $pzero) (fp.div $rne_full $random_fp $pzero))))))

}) : () -> ()
