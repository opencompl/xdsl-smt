// RUN: superoptimize %s -configuration=transfer --max-num-ops=2 --dialect=%S/apint.irdl | filecheck %s

  "func.func"() ({
  ^0(%x : !transfer.integer, %y : !transfer.integer):
    %r = "transfer.sub"(%x, %y) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %r2 = "transfer.add"(%y, %x) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %r3 = "transfer.add"(%r, %r2) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    %r4 = "transfer.add"(%r3, %x) : (!transfer.integer,!transfer.integer) -> !transfer.integer
    "func.return"(%r4) : (!transfer.integer) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer, !transfer.integer) -> !transfer.integer, "sym_visibility" = "private"} : () -> ()

// CHECK:     func.func private @test(%arg0 : !transfer.integer, %arg1 : !transfer.integer) -> !transfer.integer {
// CHECK-NEXT:  %0 = "transfer.constant"(%arg0) {value = 3 : index} : (!transfer.integer) -> !transfer.integer
// CHECK-NEXT:  %1 = "transfer.mul"(%arg0, %0) : (!transfer.integer, !transfer.integer) -> !transfer.integer
// CHECK-NEXT:  func.return %1 : !transfer.integer
// CHECK-NEXT:}

