// RUN: verify-pdl "%s" | filecheck "%s"

// 1 << 100 -> divsi 1 0
// This is incorrect as we are replacing poison with UB.
// The opposite transformation is correct

"builtin.module"() ({
  pdl.pattern @PoisonToUB : benefit(1) {
    %i32 = pdl.type : i32
    %one = pdl.attribute = 1 : i32
    %hundred = pdl.attribute = 100 : i32
    %one_op = pdl.operation "arith.constant" {"value" = %one} -> (%i32 : !pdl.type)
    %one_val = pdl.result 0 of %one_op
    %hundred_op = pdl.operation "arith.constant" {"value" = %hundred} -> (%i32 : !pdl.type)
    %hundred_val = pdl.result 0 of %hundred_op
    %1 = pdl.operation "arith.shli"(%one_val, %hundred_val : !pdl.value, !pdl.value) -> (%i32 : !pdl.type)
    pdl.rewrite %1 {
      %zero = pdl.attribute = 0 : i32
      %zero_op = pdl.operation "arith.constant" {"value" = %zero} -> (%i32 : !pdl.type)
      %zero_val = pdl.result 0 of %zero_op
      %2 = pdl.operation "arith.divsi"(%one_val, %zero_val : !pdl.value, !pdl.value) -> (%i32 : !pdl.type)
      pdl.replace %1 with %2
    }
  }
}) : () -> ()

"builtin.module"() ({
  pdl.pattern @UBToPoison : benefit(1) {
    %i32 = pdl.type : i32

    %one = pdl.attribute = 1 : i32
    %one_op = pdl.operation "arith.constant" {"value" = %one} -> (%i32 : !pdl.type)
    %one_val = pdl.result 0 of %one_op

    %zero = pdl.attribute = 0 : i32
    %zero_op = pdl.operation "arith.constant" {"value" = %zero} -> (%i32 : !pdl.type)
    %zero_val = pdl.result 0 of %zero_op

    %1 = pdl.operation "arith.divsi"(%one_val, %zero_val : !pdl.value, !pdl.value) -> (%i32 : !pdl.type)
    pdl.rewrite %1 {
      %hundred = pdl.attribute = 100 : i32
      %hundred_op = pdl.operation "arith.constant" {"value" = %hundred} -> (%i32 : !pdl.type)
      %hundred_val = pdl.result 0 of %hundred_op
      %2 = pdl.operation "arith.shli"(%one_val, %hundred_val : !pdl.value, !pdl.value) -> (%i32 : !pdl.type)

      pdl.replace %1 with %2
    }
  }
}) : () -> ()

// CHECK:      Verifying pattern PoisonToUB:
// CHECK-NEXT: with types (): UNSOUND
// CHECK-NEXT: Verifying pattern UBToPoison:
// CHECK-NEXT: with types (): SOUND
