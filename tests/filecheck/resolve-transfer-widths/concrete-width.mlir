// RUN: xdsl-smt %s -p=resolve-transfer-widths{width=8} | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer<8>):
    "func.return"(%x) : (!transfer.integer<8>) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer<8>) -> !transfer.integer<8>, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func private @test(%x : !transfer.integer<8>) -> !transfer.integer<8> {
// CHECK-NEXT:     func.return %x : !transfer.integer<8>
// CHECK-NEXT:   }
// CHECK-NEXT: }
