//https://github.com/llvm/Polygeist/blob/main/test/polygeist-opt/betterbreak2.mlir

module {
  func.func @main(%n : index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %false = arith.constant false
    %r:3 = scf.for %arg1 = %c0 to %n step %c1 iter_args(%arg2 = %c-1_i32, %arg3 = %c0_i32, %arg4 = %true) -> (i32, i32, i1) {
      %1:3 = scf.if %arg4 -> (i32, i32, i1) {
        %2 = "test.cond"() : () -> i1
        %3 = arith.select %2, %arg3, %arg2 : i32
        %4 = arith.xori %2, %true : i1
        %5 = scf.if %2 -> (i32) {
          scf.yield %arg3 : i32
        } else {
          %6 = arith.addi %arg3, %c1_i32 : i32
          scf.yield %6 : i32
        }
        scf.yield %3, %5, %4 : i32, i32, i1
      } else {
        scf.yield %arg2, %arg3, %false : i32, i32, i1
      }
      scf.yield %1#0, %1#1, %1#2 : i32, i32, i1
    }
    return %r#0 : i32
  }
}
