// RUN: xdsl-smt "%s" -p=pdl-to-smt,canonicalize-smt -t smt | filecheck "%s"

builtin.module {
    // x * -1 -> 0 - x
    pdl.pattern @mul_minus_one : benefit(0) {
        %type = pdl.type : i32

        %x = pdl.operand : %type
        %c0_attr = pdl.attribute : %type

        pdl.apply_native_constraint "is_minus_one"(%c0_attr : !pdl.attribute)

        %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%type : !pdl.type)

        %c0 = pdl.result 0 of %c0_op

        %add = pdl.operation "arith.muli" (%x, %c0 : !pdl.value, !pdl.value) -> (%type : !pdl.type)

        pdl.rewrite %add {
            %zero_attr = pdl.attribute = 0 : i32
            %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%type : !pdl.type)
            %zero = pdl.result 0 of %zero_op
            %res = pdl.operation "arith.subi"(%zero, %x : !pdl.value, !pdl.value) -> (%type : !pdl.type)
            pdl.replace %add with %res
        }
    }
}

// CHECK:       (declare-const x (Pair (_ BitVec 32) Bool))
// CHECK-NEXT:  (declare-const c0_attr (_ BitVec 32))
// CHECK-NEXT:  (assert (and (distinct (pair (bvsub (_ bv0 32) (first x)) (second x)) (pair (bvmul (first x) c0_attr) (second x))) (= c0_attr (_ bv4294967295 32))))
// CHECK-NEXT:  (check-sat)
