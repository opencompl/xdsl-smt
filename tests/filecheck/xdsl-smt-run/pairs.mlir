// RUN: xdsl-smt-run %s --args="[30, false], [12, false]" | FileCheck %s

func.func @main(%arg0: !smt.utils.pair<!smt.bv<32>, !smt.bool>, %arg1: !smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.utils.pair<!smt.bv<32>, !smt.bool> {
  %2 = "smt.utils.first"(%arg1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
  %3 = "smt.utils.second"(%arg1) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
  %4 = "smt.utils.first"(%arg0) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bv<32>
  %5 = "smt.utils.second"(%arg0) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> !smt.bool
  %6 = smt.or %3, %5
  %7 = "smt.bv.add"(%2, %4) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
  %8 = "smt.utils.pair"(%7, %6) : (!smt.bv<32>, !smt.bool) -> !smt.utils.pair<!smt.bv<32>, !smt.bool>
  "func.return"(%8) : (!smt.utils.pair<!smt.bv<32>, !smt.bool>) -> ()
}

// CHECK: (BitVectorAttr(value=IntAttr(data=42), type=BitVectorType(width=IntAttr(data=32))), False)
