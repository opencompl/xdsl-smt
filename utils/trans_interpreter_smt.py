from typing import Callable

from xdsl.dialects.func import FuncOp, Return, Call
from xdsl.ir import Operation
import dialects.transfer as transfer

"""
Interpret a function from transfer dialect in MLIR to a Python function.
The parsed function receives SMT variables and a solver, returns expected SMT value
Function call should be recursive parsed
"""

INDENT = "  "

DEBUG = False


def wrap_operation(op_str: str):
    return INDENT + op_str + "\n"


# wrap function body with signature
def wrap_function_body(body: str, func: FuncOp) -> str:
    parameters = "solver"
    for i in range(0, len(func.args)):
        arg_name = func.args[i].name_hint
        assert arg_name is not None
        parameters += "," + arg_name
    parameters = "(" + parameters + ")"
    signature = "def " + func.sym_name.data + parameters + ":\n"
    return signature + body


def get_smt_function(op: Operation) -> str:
    if isinstance(op, transfer.CmpOp):
        predicate = op.predicate.value.data
        return "opToSMTFunc['" + op.name + "'][" + str(predicate) + "]"
    return "opToSMTFunc['" + op.name + "']"


def handle_op_attributes(op: Operation, operands: str) -> str:
    if isinstance(op, transfer.Constant):
        operands = operands + ", " + str(op.value.value.data)
    elif isinstance(op, transfer.GetOp):
        operands = operands + ", " + str(op.index.value.data)
    return operands


def parse_function_to_python(
    func: FuncOp,
    func_name_to_func: dict[str, FuncOp],  # type: ignore
    op_to_smt_func: dict[str, Callable],  # type: ignore
    func_call: dict[str, Callable],  # type: ignore
) -> Callable:  # type: ignore
    body = ""
    function_name = func.sym_name.data
    blk = func.body.blocks[0]
    for op in blk.ops:
        if isinstance(op, Return):
            arg_name = op.arguments[0].name_hint
            assert arg_name is not None
            op_str = "return " + arg_name
            op_str = wrap_operation(op_str)
            body += op_str
            break
        returned_value = op.results[0].name_hint
        assert returned_value is not None
        if not isinstance(op, Call):
            smt_func = get_smt_function(op)
        else:
            callee = op.callee.string_value()
            if callee not in func_call:
                assert callee in func_name_to_func
                callee_func = func_name_to_func[callee]
                parse_function_to_python(
                    callee_func, func_name_to_func, op_to_smt_func, func_call
                )
            smt_func = "funcCall['" + callee + "']"
        operands = "solver"
        for i in range(0, len(op.operands)):
            arg_name = op.operands[i].name_hint
            assert arg_name is not None
            operands += "," + arg_name
        operands = handle_op_attributes(op, operands)
        operands = "(" + operands + ")"
        op_str = returned_value + "=" + smt_func + operands
        op_str = wrap_operation(op_str)
        body += op_str
        if DEBUG:
            debug = wrap_operation("print('" + op.name + "')")
            body += debug
            debug = wrap_operation("print" + operands)
            body += debug
            debug = wrap_operation(
                "print(" + "'" + returned_value + "'," + returned_value + ")"
            )
            body += debug

    whole_function = wrap_function_body(body, func)
    if DEBUG:
        print(whole_function)
    local = {}
    gbl = {"opToSMTFunc": op_to_smt_func, "funcCall": func_call}  # type: ignore
    exec(whole_function, gbl, local)  # type: ignore
    func_obj: Callable = local[function_name]  # type: ignore
    func_call[function_name] = func_obj  # type: ignore
    return func_obj  # type: ignore
