// RUN: superoptimize %s --max-num-ops=2 | filecheck %s

func.func @foo(%x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %r = arith.muli %x, %c0 : i32
  func.return %r : i32
}

// CHECK:      func.func @foo(%arg0 : i32) -> i32 attributes {seed = 1 : index} {
// CHECK-NEXT:   %0 = "hw.constant"() {value = 0 : i32} : () -> i32
// CHECK-NEXT:   func.return %0 : i32
// CHECK-NEXT: }
