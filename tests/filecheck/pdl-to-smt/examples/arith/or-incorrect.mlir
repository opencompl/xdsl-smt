// RUN: xdsl-smt "%s" -p=pdl-to-smt -t smt | z3 -in | filecheck "%s"

// or(x, y) -> y

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.type"() {constantType = i32} : () -> !pdl.type
    %1 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %2 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %3 = pdl.operation "arith.ori"(%1, %2 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
    pdl.rewrite %3 {
      pdl.replace %3 with (%2 : !pdl.value)
    }
  }) {benefit = 1 : i16} : () -> ()
}) : () -> ()


// CHECK-NOT: unsat
