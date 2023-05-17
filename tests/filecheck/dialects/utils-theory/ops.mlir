// RUN: xdsl-smt.py %s | xdsl-smt.py -t=smt | filecheck %s
// RUN: xdsl-smt.py %s -t=smt | z3 -in

"builtin.module"() ({
    %x = "smt.declare_const"() : () -> !smt.utils.pair<!smt.bool, !smt.bool>
    // CHECK:      (declare-const x (Pair Bool Bool))

    %first = "smt.utils.first"(%x) : (!smt.utils.pair<!smt.bool, !smt.bool>) -> !smt.bool
    "smt.assert"(%first) : (!smt.bool) -> ()
    // CHECK-NEXT: (assert (first x))

    %second = "smt.utils.second"(%x) : (!smt.utils.pair<!smt.bool, !smt.bool>) -> !smt.bool
    "smt.assert"(%second) : (!smt.bool) -> ()
    // CHECK-NEXT: (assert (second x))

    %pair = "smt.utils.pair"(%first, %second) : (!smt.bool, !smt.bool) -> !smt.utils.pair<!smt.bool, !smt.bool>
    %eq = "smt.eq"(%pair, %x) : (!smt.utils.pair<!smt.bool, !smt.bool>, !smt.utils.pair<!smt.bool, !smt.bool>) -> !smt.bool
    "smt.assert"(%eq) : (!smt.bool) -> ()
    // CHECK-NEXT: (assert (= (pair (first x) (second x)) x))
}): () -> ()