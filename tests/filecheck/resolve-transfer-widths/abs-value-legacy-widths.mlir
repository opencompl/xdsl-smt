// RUN: xdsl-smt %s -p=resolve-transfer-widths{width=8} | filecheck %s

builtin.module {
  func.func @abst_xfer(%lhs : !transfer.abs_value<[!transfer.integer, !transfer.integer]>, %rhs : !transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]> {
    %lhs0 = "transfer.get"(%lhs) {index = 0 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
    %lhs1 = "transfer.get"(%lhs) {index = 1 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
    %rhs0 = "transfer.get"(%rhs) {index = 0 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
    %rhs1 = "transfer.get"(%rhs) {index = 1 : index} : (!transfer.abs_value<[!transfer.integer, !transfer.integer]>) -> !transfer.integer
    %res = "transfer.make"(%lhs0, %rhs1) : (!transfer.integer, !transfer.integer) -> !transfer.abs_value<[!transfer.integer, !transfer.integer]>
    func.return %res : !transfer.abs_value<[!transfer.integer, !transfer.integer]>
  }
}

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func @abst_xfer(%lhs : !transfer.abs_value<[!transfer.integer<8>, !transfer.integer<8>]>, %rhs : !transfer.abs_value<[!transfer.integer<8>, !transfer.integer<8>]>) -> !transfer.abs_value<[!transfer.integer<8>, !transfer.integer<8>]> {
// CHECK-NEXT:     %lhs0 = "transfer.get"(%lhs) {index = 0 : index} : (!transfer.abs_value<[!transfer.integer<8>, !transfer.integer<8>]>) -> !transfer.integer<8>
// CHECK-NEXT:     %lhs1 = "transfer.get"(%lhs) {index = 1 : index} : (!transfer.abs_value<[!transfer.integer<8>, !transfer.integer<8>]>) -> !transfer.integer<8>
// CHECK-NEXT:     %rhs0 = "transfer.get"(%rhs) {index = 0 : index} : (!transfer.abs_value<[!transfer.integer<8>, !transfer.integer<8>]>) -> !transfer.integer<8>
// CHECK-NEXT:     %rhs1 = "transfer.get"(%rhs) {index = 1 : index} : (!transfer.abs_value<[!transfer.integer<8>, !transfer.integer<8>]>) -> !transfer.integer<8>
// CHECK-NEXT:     %res = "transfer.make"(%lhs0, %rhs1) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.abs_value<[!transfer.integer<8>, !transfer.integer<8>]>
// CHECK-NEXT:     func.return %res : !transfer.abs_value<[!transfer.integer<8>, !transfer.integer<8>]>
// CHECK-NEXT:   }
// CHECK-NEXT: }
