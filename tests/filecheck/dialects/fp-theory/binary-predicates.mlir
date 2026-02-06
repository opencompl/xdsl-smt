// RUN: xdsl-smt "%s" | xdsl-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt "%s" -t=smt | z3 -in

"builtin.module"() ({

  %rne_full = "smt.fp.round_nearest_ties_to_even"() : () -> !smt.fp.rounding_mode
  %three = smt.bv.constant #smt.bv<3> : !smt.bv<8>
  %four = smt.bv.constant #smt.bv<4> : !smt.bv<10>
  %zero = smt.bv.constant #smt.bv<0> : !smt.bv<1>
  %random_fp = "smt.fp.constant"(%zero, %three, %four) : (!smt.bv<1>, !smt.bv<8>, !smt.bv<10>) -> !smt.fp<8,11>
  %pzero = "smt.fp.pzero"() : () -> !smt.fp<8,11>

  %eq = "smt.fp.eq"(%random_fp, %pzero) : (!smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.bool
  %lt = "smt.fp.lt"(%random_fp, %pzero) : (!smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.bool
  %leq = "smt.fp.leq"(%random_fp, %pzero) : (!smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.bool
  %gt = "smt.fp.gt"(%random_fp, %pzero) : (!smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.bool
  %geq = "smt.fp.geq"(%random_fp, %pzero) : (!smt.fp<8, 11>, !smt.fp<8, 11>) -> !smt.bool

  "smt.assert"(%eq) : (!smt.bool) -> ()
  "smt.assert"(%lt) : (!smt.bool) -> ()
  "smt.assert"(%leq) : (!smt.bool) -> ()
  "smt.assert"(%gt) : (!smt.bool) -> ()
  "smt.assert"(%geq) : (!smt.bool) -> ()

  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK:   (let (($pzero (_ +zero 8 11)))
  // CHECK:   (fp.eq $random_fp $pzero))))
  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK:   (let (($pzero (_ +zero 8 11)))
  // CHECK:   (fp.lt $random_fp $pzero))))
  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK:   (let (($pzero (_ +zero 8 11)))
  // CHECK:   (fp.leq $random_fp $pzero))))
  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK:   (let (($pzero (_ +zero 8 11)))
  // CHECK:   (fp.gt $random_fp $pzero))))
  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK:   (let (($pzero (_ +zero 8 11)))
  // CHECK:   (fp.geq $random_fp $pzero))))
}) : () -> ()
