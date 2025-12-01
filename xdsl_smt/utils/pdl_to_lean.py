"""
Converts PDL patterns to Lean theorem statements.
"""

from xdsl.dialects import pdl, smt
from xdsl.ir import SSAValue
from xdsl_smt.dialects.smt_bitvector_dialect import AddOp


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


op_name_to_lean_infix: dict[str, str] = {
    "smt.eq": "=",
    "smt.distinct": "≠",
    "smt.bv.sub": "-",
    "smt.bv.mul": "*",
    "smt.bv.add": "+",
}


def get_lean_expression(value: SSAValue, operand_to_str: dict[SSAValue, str]) -> str:
    if value in operand_to_str:
        return operand_to_str[value]
    if isinstance((pdl_op := value.owner), pdl.OperationOp):
        assert pdl_op.opName is not None
        op_name = pdl_op.opName.data
        lean_infix = op_name_to_lean_infix[op_name]
        lhs = get_lean_expression(pdl_op.operand_values[0], operand_to_str)
        rhs = get_lean_expression(pdl_op.operand_values[1], operand_to_str)
        return f"({lhs} {lean_infix} {rhs})"
    assert False, f"Unsupported owner: {value.owner}"


def get_lean_type(value: SSAValue) -> str:
    assert isinstance(value.owner, pdl.TypeOp)
    assert (MlirType := value.owner.constantType) is not None
    assert isinstance(MlirType, smt.BitVectorType)
    return f"BitVec {MlirType.width}"


def pdl_to_lean(pattern: pdl.PatternOp) -> str:
    operand_to_str: dict[SSAValue, str] = {}
    operand_to_type: dict[SSAValue, str] = {}
    # Support up to 7 operands for now
    names = ["a", "b", "c", "d", "e", "f", "g"]
    num_names = 0
    for op in pattern.walk():
        if isinstance(op, pdl.OperandOp):
            operand_to_str[op.value] = names[num_names]
            operand_to_type[op.value] = get_lean_type(op.value)
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
    assert pattern.sym_name is not None
    arguments_str = " ".join(
        f"({operand_to_str[value]} : {operand_to_type[value]})"
        for value in operand_to_str.keys()
    )
    return f"theorem {pattern.sym_name} {arguments_str}:\n    {lhs_expr} = {rhs_expr} := by\n  sorry"
