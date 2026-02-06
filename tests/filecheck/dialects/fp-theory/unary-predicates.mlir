// RUN: xdsl-smt "%s" | xdsl-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt "%s" -t=smt | z3 -in

"builtin.module"() ({

  %rne_full = "smt.fp.round_nearest_ties_to_even"() : () -> !smt.fp.rounding_mode
  %three = smt.bv.constant #smt.bv<3> : !smt.bv<8>
  %four = smt.bv.constant #smt.bv<4> : !smt.bv<10>
  %zero = smt.bv.constant #smt.bv<0> : !smt.bv<1>
  %random_fp = "smt.fp.constant"(%zero, %three, %four) : (!smt.bv<1>, !smt.bv<8>, !smt.bv<10>) -> !smt.fp<8,11>
  %pzero = "smt.fp.pzero"() : () -> !smt.fp<8,11>

  %isNormal = "smt.fp.isNormal"(%random_fp) : (!smt.fp<8, 11>) -> !smt.bool
  %isSubnormal = "smt.fp.isSubnormal"(%random_fp) : (!smt.fp<8, 11>) -> !smt.bool
  %isZero = "smt.fp.isZero"(%random_fp) : (!smt.fp<8, 11>) -> !smt.bool
  %isInfinite = "smt.fp.isInfinite"(%random_fp) : (!smt.fp<8, 11>) -> !smt.bool
  %isNaN = "smt.fp.isNaN"(%random_fp) : (!smt.fp<8, 11>) -> !smt.bool
  %isNegative = "smt.fp.isNegative"(%random_fp) : (!smt.fp<8, 11>) -> !smt.bool
  %isPositive = "smt.fp.isPositive"(%random_fp) : (!smt.fp<8, 11>) -> !smt.bool


  "smt.assert"(%isNormal) : (!smt.bool) -> ()
  "smt.assert"(%isSubnormal) : (!smt.bool) -> ()
  "smt.assert"(%isZero) : (!smt.bool) -> ()
  "smt.assert"(%isInfinite) : (!smt.bool) -> ()
  "smt.assert"(%isNaN) : (!smt.bool) -> ()
  "smt.assert"(%isNegative) : (!smt.bool) -> ()
  "smt.assert"(%isPositive) : (!smt.bool) -> ()
  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK:   (fp.isNormal $random_fp)))
  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK:   (fp.isSubnormal $random_fp)))
  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK:   (fp.isZero $random_fp)))
  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK: (fp.isInfinite $random_fp)))
  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK: (fp.isNaN $random_fp)))
  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK: (fp.isNegative $random_fp)))
  // CHECK: (assert (let (($random_fp (fp (_ bv0 1) (_ bv3 8) (_ bv4 10))))
  // CHECK: (fp.isPositive $random_fp)))
}) : () -> ()
