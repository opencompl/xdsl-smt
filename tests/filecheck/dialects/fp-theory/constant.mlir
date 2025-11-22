// RUN: xdsl-smt "%s" | xdsl-smt -t=smt | filecheck "%s"
// RUN: xdsl-smt "%s" -t=smt | z3 -in

"builtin.module"() ({

  %three = smt.bv.constant #smt.bv<3> : !smt.bv<8>
  %four = smt.bv.constant #smt.bv<4> : !smt.bv<10>
  %zero = smt.bv.constant #smt.bv<0> : !smt.bv<1>
  %random_fp = "smt.fp.constant"(%zero, %three, %four) : (!smt.bv<1>, !smt.bv<8>, !smt.bv<10>) -> !smt.fp<8,11>
  %pzero = "smt.fp.pzero"() : () -> !smt.fp<8,11>
  %nzero = "smt.fp.nzero"() : () -> !smt.fp<8,11>
  %ninf = "smt.fp.ninf"() : () -> !smt.fp<8,11>
  %pinf = "smt.fp.pinf"() : () -> !smt.fp<8,11>
  %nan = "smt.fp.nan"() : () -> !smt.fp<8,11>

  %eq_inf = "smt.eq"(%pinf, %ninf) : (!smt.fp<8,11>, !smt.fp<8,11>) -> !smt.bool
  "smt.assert"(%eq_inf) : (!smt.bool) -> ()
  // CHECK: (assert (= (_ +oo 8 11) (_ -oo 8 11)))

  %eq_zero = "smt.eq"(%pzero, %nzero) : (!smt.fp<8,11>, !smt.fp<8,11>) -> !smt.bool
  "smt.assert"(%eq_zero) : (!smt.bool) -> ()
  // CHECK: (assert (= (_ +zero 8 11) (_ -zero 8 11)))

  %eq = "smt.eq"(%random_fp, %nan) : (!smt.fp<8,11>, !smt.fp<8,11>) -> !smt.bool
  "smt.assert"(%eq) : (!smt.bool) -> ()
  // CHECK: (assert (= (fp (_ bv0 1) (_ bv3 8) (_ bv4 10)) (_ NaN 8 11)))



}) : () -> ()
