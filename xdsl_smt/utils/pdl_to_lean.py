"""
Converts PDL patterns to Lean theorem statements.
"""

from xdsl.dialects import pdl, smt
from xdsl.ir import SSAValue
from xdsl.context import Context
from xdsl_smt.dialects import get_all_dialects


def get_replace_terminator(pattern: pdl.PatternOp) -> pdl.ReplaceOp:
    """
    Get the pdl.replace operation that terminates the rewrite block of a PDL pattern.
    """
    rewrite = pattern.body.block.last_op
    assert isinstance(rewrite, pdl.RewriteOp)
    assert rewrite.body is not None
    replace = rewrite.body.block.last_op
    assert isinstance(replace, pdl.ReplaceOp)
    return replace


unary_lean_name: dict[str, str] = {
    "smt.bv.neg": "-",
    "smt.bv.not": "BitVec.not",
}

op_name_to_lean_infix: dict[str, str] = {
    "smt.eq": "=",
    "smt.distinct": "≠",
    "smt.bv.sub": "-",
    "smt.bv.mul": "*",
    "smt.bv.add": "+",
    "smt.bv.and": "&&&",
    "smt.bv.or": "|||",
    "smt.bv.xor": "^^^",
    "smt.bv.shl": "<<<",
}

op_name_to_lean_prefix: dict[str, str] = {
    "smt.bv.ashr": "BitVec.sshiftRight'",
}


def get_lean_expression(value: SSAValue, operand_to_str: dict[SSAValue, str]) -> str:
    if value in operand_to_str:
        return operand_to_str[value]
    if isinstance((pdl_op := value.owner), pdl.ResultOp):
        return get_lean_expression(pdl_op.parent_, operand_to_str)
    if isinstance((pdl_op := value.owner), pdl.OperationOp):
        operands = [
            get_lean_expression(operand, operand_to_str)
            for operand in pdl_op.operand_values
        ]
        assert pdl_op.opName is not None
        op_name = pdl_op.opName.data
        if len(operands) == 1:
            return f"({unary_lean_name[op_name]} {operands[0]})"
        elif len(operands) == 2:
            lhs, rhs = operands
            if op_name in op_name_to_lean_infix.keys():
                lean_infix = op_name_to_lean_infix[op_name]
                return f"({lhs} {lean_infix} {rhs})"
            if op_name in op_name_to_lean_prefix.keys():
                lean_prefix = op_name_to_lean_prefix[op_name]
                return f"({lean_prefix} {lhs} {rhs})"
            if op_name == "smt.bv.lshr":
                return f"(BitVec.ushiftRight {lhs} {rhs}.toNat)"
            assert False, f"Unsupported operation: {op_name}"
    assert False, f"Unsupported owner: {value.owner}"


def get_lean_type(value: SSAValue) -> str:
    if not isinstance(value.owner, pdl.TypeOp):
        raise ValueError(f"Only pdl.type values are supported, got {value.owner}")
    assert (MlirType := value.owner.constantType) is not None
    assert isinstance(MlirType, smt.BitVectorType)
    return f"BitVec {MlirType.width.data}"


def pdl_to_lean(pattern: pdl.PatternOp) -> str:
    operand_to_str: dict[SSAValue, str] = {}
    operand_to_type: dict[SSAValue, str] = {}
    # Support up to 7 operands for now
    names = ["a", "b", "c", "d", "e", "f", "g"]
    num_names = 0
    for op in pattern.walk():
        if isinstance(op, pdl.OperandOp):
            operand_to_str[op.value] = names[num_names]
            assert op.value_type is not None
            operand_to_type[op.value] = get_lean_type(op.value_type)
            num_names += 1

    replace_op = get_replace_terminator(pattern)
    lhs_expr = get_lean_expression(replace_op.op_value, operand_to_str)
    if replace_op.repl_operation is not None:
        rhs_expr = get_lean_expression(replace_op.repl_operation, operand_to_str)
    else:
        assert replace_op.repl_values is not None
        assert len(replace_op.repl_values) == 1
        rhs_expr = get_lean_expression(replace_op.repl_values[0], operand_to_str)

    # Convert the PDL pattern to a Lean theorem statement.
    # assert pattern.sym_name is not None
    arguments_str = " ".join(
        f"({operand_to_str[value]} : {operand_to_type[value]})"
        for value in operand_to_str.keys()
    )
    return f"example {arguments_str}:\n    {lhs_expr} = {rhs_expr} := by\n  bv_decide +acNf"


if __name__ == "__main__":
    import sys

    from xdsl.parser import Parser

    if len(sys.argv) != 2:
        print("Usage: pdl_to_lean.py <pdl_file>")
        sys.exit(1)

    pdl_file = sys.argv[1]
    with open(pdl_file, "r") as f:
        pdl_text = f.read()

    ctx = Context()
    for name, factory in get_all_dialects().items():
        ctx.register_dialect(name, factory)

    module = Parser(ctx, pdl_text).parse_module()

    for pattern in module.walk():
        if isinstance(pattern, pdl.PatternOp):
            lean_theorem = pdl_to_lean(pattern)
            # aprint(pattern)
            print(lean_theorem)
