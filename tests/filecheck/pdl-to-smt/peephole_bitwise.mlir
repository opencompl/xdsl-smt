// RUN: xdsl-smt -p=pdl-to-smt,lower-effects,canonicalize,dce -t smt "%s" | z3 -in
pdl.pattern : benefit(1) {
    %in_type = pdl.type: !transfer.integer
    %x = pdl.operand : %in_type
    %y = pdl.operand : %in_type
    %andi_op = pdl.operation "arith.andi" (%x,%y: !pdl.value, !pdl.value)  -> (%in_type: !pdl.type)
    %andi = pdl.result 0 of %andi_op
    %xori_op = pdl.operation "arith.xori" (%x,%y: !pdl.value, !pdl.value)  -> (%in_type: !pdl.type)
    %xori = pdl.result 0 of %xori_op
    %root_addi = pdl.operation "arith.addi" (%andi,%xori: !pdl.value, !pdl.value)  -> (%in_type: !pdl.type)
    pdl.rewrite %root_addi {
      %ori_op = pdl.operation "arith.ori" (%x,%y: !pdl.value, !pdl.value)  -> (%in_type: !pdl.type)
      pdl.replace %root_addi with %ori_op
     }
}

//CHECK: unsat
