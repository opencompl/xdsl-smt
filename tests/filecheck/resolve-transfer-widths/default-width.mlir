// RUN: xdsl-smt %s -p=resolve-transfer-widths | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer):
    "func.return"(%x) : (!transfer.integer) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer) -> !transfer.integer, "sym_visibility" = "private"} : () -> ()
}) {"transfer.default_width" = 8 : index} : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func private @test(%x : !transfer.integer<8>) -> !transfer.integer<8> {
// CHECK-NEXT:     func.return %x : !transfer.integer<8>
// CHECK-NEXT:   }
// CHECK-NEXT: }
