// RUN: xdsl-smt %s -p=resolve-transfer-widths{width-map=\"@X=8,@Y=16\"} | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer<@X>, %y : !transfer.integer<@X>):
    %r = "transfer.add"(%x, %y) : (!transfer.integer<@X>, !transfer.integer<@X>) -> !transfer.integer<@X>
    "func.return"(%r) : (!transfer.integer<@X>) -> ()
  }) {"sym_name" = "test_x", "function_type" = (!transfer.integer<@X>, !transfer.integer<@X>) -> !transfer.integer<@X>, "sym_visibility" = "private"} : () -> ()

  "func.func"() ({
  ^0(%x : !transfer.integer<@Y>):
    "func.return"(%x) : (!transfer.integer<@Y>) -> ()
  }) {"sym_name" = "test_y", "function_type" = (!transfer.integer<@Y>) -> !transfer.integer<@Y>, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func private @test_x(%x : !transfer.integer<8>, %y : !transfer.integer<8>) -> !transfer.integer<8> {
// CHECK-NEXT:     %r = "transfer.add"(%x, %y) : (!transfer.integer<8>, !transfer.integer<8>) -> !transfer.integer<8>
// CHECK-NEXT:     func.return %r : !transfer.integer<8>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @test_y(%x : !transfer.integer<16>) -> !transfer.integer<16> {
// CHECK-NEXT:     func.return %x : !transfer.integer<16>
// CHECK-NEXT:   }
// CHECK-NEXT: }
