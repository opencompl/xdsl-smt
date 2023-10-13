// RUN: xdsl-smt "%s" -p=pdl-to-smt -t smt | z3 -in | filecheck "%s"

// or(x, or(x, y)) -> or(x, y)

"builtin.module"() ({
  "pdl.pattern"() ({
    %0 = "pdl.type"() {constantType = i32} : () -> !pdl.type
    %1 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %2 = "pdl.operand"(%0) : (!pdl.type) -> !pdl.value
    %3 = pdl.operation "arith.ori"(%1, %2 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
    %4 = "pdl.result"(%3) {index = 0 : i32} : (!pdl.operation) -> !pdl.value
    %5 = pdl.operation "arith.ori"(%1, %4 : !pdl.value, !pdl.value) -> (%0 : !pdl.type)
    pdl.rewrite %5 {
      pdl.replace %5 with (%4 : !pdl.value)
    }
  }) {benefit = 1 : i16} : () -> ()
}) : () -> ()


// CHECK: unsat
