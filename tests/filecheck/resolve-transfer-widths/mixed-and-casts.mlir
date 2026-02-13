// RUN: xdsl-smt %s -p=resolve-transfer-widths{width-map=\"@X=8,@Y=16\"} | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer<@X>, %y : !transfer.integer<16>):
    %z = "transfer.zext"(%x) : (!transfer.integer<@X>) -> !transfer.integer<@Y>
    %t = "transfer.trunc"(%y) : (!transfer.integer<16>) -> !transfer.integer<@X>
    %s = "transfer.sext"(%t) : (!transfer.integer<@X>) -> !transfer.integer<@Y>
    "func.return"(%s) : (!transfer.integer<@Y>) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer<@X>, !transfer.integer<16>) -> !transfer.integer<@Y>, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:      builtin.module {
// CHECK-NEXT:   func.func private @test(%x : !transfer.integer<8>, %y : !transfer.integer<16>) -> !transfer.integer<16> {
// CHECK-NEXT:     %z = "transfer.zext"(%x) : (!transfer.integer<8>) -> !transfer.integer<16>
// CHECK-NEXT:     %t = "transfer.trunc"(%y) : (!transfer.integer<16>) -> !transfer.integer<8>
// CHECK-NEXT:     %s = "transfer.sext"(%t) : (!transfer.integer<8>) -> !transfer.integer<16>
// CHECK-NEXT:     func.return %s : !transfer.integer<16>
// CHECK-NEXT:   }
// CHECK-NEXT: }
