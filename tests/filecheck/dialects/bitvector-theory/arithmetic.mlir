// RUN: xdsl-smt.py %s | xdsl-smt.py -t=smt | filecheck %s
// RUN: xdsl-smt.py %s -t=smt | z3 -in


"builtin.module"() ({
  %x = "smt.declare_const"() : () -> !smt.bv.bv<32>
  %y = "smt.declare_const"() : () -> !smt.bv.bv<32>
  %z = "smt.declare_const"() : () -> !smt.bv.bv<32>
  // CHECK:      (declare-const x (_ BitVec 32))
  // CHECK-NEXT: (declare-const y (_ BitVec 32))
  // CHECK-NEXT: (declare-const z (_ BitVec 32))

  %neg = "smt.bv.neg"(%x) : (!smt.bv.bv<32>) -> !smt.bv.bv<32>
  %eq_neg = "smt.eq"(%z, %neg) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%eq_neg) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= z (bvneg x)))

  %add = "smt.bv.add"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
  %eq_add = "smt.eq"(%z, %add) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%eq_add) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= z (bvadd x y)))

  %sub = "smt.bv.sub"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
  %eq_sub = "smt.eq"(%z, %sub) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%eq_sub) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= z (bvsub x y)))

  %mul = "smt.bv.mul"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
  %eq_mul = "smt.eq"(%z, %mul) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%eq_mul) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= z (bvmul x y)))

  %urem = "smt.bv.urem"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
  %eq_urem = "smt.eq"(%z, %urem) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%eq_urem) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= z (bvurem x y)))

  %srem = "smt.bv.srem"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
  %eq_srem = "smt.eq"(%z, %srem) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%eq_srem) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= z (bvsrem x y)))

  %smod = "smt.bv.smod"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
  %eq_smod = "smt.eq"(%z, %smod) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%eq_smod) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= z (bvsmod x y)))

  %shl = "smt.bv.shl"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
  %eq_shl = "smt.eq"(%z, %shl) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%eq_shl) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= z (bvshl x y)))

  %lshr = "smt.bv.lshr"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
  %eq_lshr = "smt.eq"(%z, %lshr) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%eq_lshr) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= z (bvlshr x y)))

  %ashr = "smt.bv.ashr"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
  %eq_ashr = "smt.eq"(%z, %ashr) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%eq_ashr) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= z (bvashr x y)))

  %udiv = "smt.bv.udiv"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
  %eq_udiv = "smt.eq"(%z, %udiv) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%eq_udiv) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= z (bvudiv x y)))

  %sdiv = "smt.bv.sdiv"(%x, %y) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bv.bv<32>
  %eq_sdiv = "smt.eq"(%z, %sdiv) : (!smt.bv.bv<32>, !smt.bv.bv<32>) -> !smt.bool
  "smt.assert"(%eq_sdiv) : (!smt.bool) -> ()
  // CHECK-NEXT: (assert (= z (bvsdiv x y)))
}) : () -> ()
