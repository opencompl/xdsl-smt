// RUN: xdsl-smt %s -p=resolve-transfer-widths{width-map=\"@X=8,@Y=16\"},lower-to-smt,lower-effects,canonicalize,dce,merge-func-results -t smt | filecheck %s

"builtin.module"() ({
  "func.func"() ({
  ^0(%x : !transfer.integer<@X>, %y : !transfer.integer<@Y>):
    %zx = "transfer.zext"(%x) : (!transfer.integer<@X>) -> !transfer.integer<@Y>
    %ty = "transfer.trunc"(%y) : (!transfer.integer<@Y>) -> !transfer.integer<@X>
    %add = "transfer.add"(%ty, %x) : (!transfer.integer<@X>, !transfer.integer<@X>) -> !transfer.integer<@X>
    %sx = "transfer.sext"(%add) : (!transfer.integer<@X>) -> !transfer.integer<@Y>
    %mix = "transfer.add"(%zx, %sx) : (!transfer.integer<@Y>, !transfer.integer<@Y>) -> !transfer.integer<@Y>
    "func.return"(%mix) : (!transfer.integer<@Y>) -> ()
  }) {"sym_name" = "test", "function_type" = (!transfer.integer<@X>, !transfer.integer<@Y>) -> !transfer.integer<@Y>, "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// CHECK:       (define-fun $test
// CHECK-SAME:  (_ BitVec 8)
// CHECK-SAME:  (_ BitVec 16)
// CHECK-DAG:   pair (bvadd
// CHECK-DAG:   (_ zero_extend 8)
// CHECK-DAG:   (_ sign_extend 8) (bvadd ((_ extract 7 0)
