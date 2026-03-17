// RUN: xdsl-smt %s -p=resolve-transfer-widths{width=8} | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer<16>):
    %r = "transfer.neg"(%x) : (!transfer.integer) -> !transfer.integer
    "func.return"(%r) : (!transfer.integer) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer, !transfer.integer<16>) -> !transfer.integer, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func private @test(%x : !transfer.integer<8>, %y : !transfer.integer<16>) -> !transfer.integer<8> {
// CHECK-NEXT:     %r = "transfer.neg"(%x) : (!transfer.integer<8>) -> !transfer.integer<8>
// CHECK-NEXT:     func.return %r : !transfer.integer<8>
// CHECK-NEXT:   }
// CHECK-NEXT: }
