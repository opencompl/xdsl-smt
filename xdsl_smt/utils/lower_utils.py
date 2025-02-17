from ..dialects.transfer import (
    AbstractValueType,
    GetOp,
    MakeOp,
    NegOp,
    Constant,
    CmpOp,
    AndOp,
    OrOp,
    XorOp,
    AddOp,
    SubOp,
    CountLOneOp,
    CountLZeroOp,
    CountROneOp,
    CountRZeroOp,
    SetHighBitsOp,
    SetLowBitsOp,
    GetLowBitsOp,
    GetBitWidthOp,
    UMulOverflowOp,
    SMinOp,
    SMaxOp,
    UMinOp,
    UMaxOp,
    TransIntegerType,
    ShlOp,
    AShrOp,
    LShrOp,
    ExtractOp,
    ConcatOp,
    GetAllOnesOp,
    SelectOp,
    NextLoopOp,
    ConstRangeForOp,
    RepeatOp,
    IntersectsOp,
    # FromArithOp,
    TupleType,
    AddPoisonOp,
    RemovePoisonOp,
)
from xdsl.dialects.func import FuncOp, ReturnOp, CallOp
from xdsl.pattern_rewriter import *
from functools import singledispatch
from typing import TypeVar, cast
from xdsl.dialects.builtin import Signedness, IntegerType, IndexType, IntegerAttr
from xdsl.ir import Operation
import xdsl.dialects.arith as arith

operNameToCpp = {
    "transfer.and": "&",
    "arith.andi": "&",
    "transfer.add": "+",
    "arith.constant": "APInt",
    "arith.addi": "+",
    "transfer.or": "|",
    "arith.ori": "|",
    "transfer.xor": "^",
    "arith.xori": "^",
    "transfer.sub": "-",
    "arith.subi": "-",
    "transfer.neg": "~",
    "transfer.mul": "*",
    "transfer.umul_overflow": ".umul_ov",
    "transfer.get_bit_width": ".getBitWidth",
    "transfer.countl_zero": ".countl_zero",
    "transfer.countr_zero": ".countr_zero",
    "transfer.countl_one": ".countl_one",
    "transfer.countr_one": ".countr_one",
    "transfer.get_low_bits": ".getLoBits",
    "transfer.set_high_bits": ".setHighBits",
    "transfer.set_low_bits": ".setLowBits",
    "transfer.intersects": ".intersects",
    "transfer.cmp": [
        ".eq",
        ".ne",
        ".slt",
        ".sle",
        ".sgt",
        ".sge",
        ".ult",
        ".ule",
        ".ugt",
        ".uge",
    ],
    # "transfer.fromArith": "APInt",
    "transfer.make": "{{{0}}}",
    "transfer.get": "[{0}]",
    "transfer.shl": ".shl",
    "transfer.ashr": ".ashr",
    "transfer.lshr": ".lshr",
    "transfer.concat": ".concat",
    "transfer.extract": ".extractBits",
    "transfer.umin": [".ule", "?", ":"],
    "transfer.smin": [".sle", "?", ":"],
    "transfer.umax": [".ugt", "?", ":"],
    "transfer.smax": [".sgt", "?", ":"],
    "func.return": "return",
    "transfer.constant": "APInt",
    "arith.select": ["?", ":"],
    "arith.cmpi": ["==", "!=", "<", "<=", ">", ">="],
    "transfer.get_all_ones": "APInt::getAllOnes",
    "transfer.select": ["?", ":"],
    "transfer.reverse_bits": ".reverseBits",
    "transfer.add_poison": " ",
    "transfer.remove_poison": " ",
    "comb.add": "+",
    "comb.sub": "-",
    "comb.mul": "*",
    "comb.and": "&",
    "comb.or": "|",
    "comb.xor": "^",
    "comb.divs": ".sdiv",
    "comb.divu": ".udiv",
    "comb.mods": ".srem",
    "comb.modu": ".urem",
    "comb.mux": ["?", ":"],
    "comb.shrs": ".alhr",
    "comb.shru": ".lshr",
    "comb.shl": ".shl",
    "comb.extract": ".extractBits",
    "comb.concat": ".concat",
}
# transfer.constRangeLoop and NextLoop are controller operations, should be handle specially

unsignedReturnedType = {
    CountLOneOp,
    CountLZeroOp,
    CountROneOp,
    CountRZeroOp,
    GetBitWidthOp,
}

ends = ";\n"
indent = "\t"
int_to_apint = False


def set_int_to_apint(to_apint: bool):
    global int_to_apint
    int_to_apint = to_apint


def lowerType(typ, specialOp=None):
    if specialOp is not None:
        for op in unsignedReturnedType:
            if isinstance(specialOp, op):
                return "unsigned"
    if isinstance(typ, TransIntegerType):
        return "APInt"
    elif isinstance(typ, AbstractValueType) or isinstance(typ, TupleType):
        fields = typ.get_fields()
        typeName = lowerType(fields[0])
        for i in range(1, len(fields)):
            assert lowerType(fields[i]) == typeName
        return "std::vector<" + typeName + ">"
    elif isinstance(typ, IntegerType):
        return "int" if not int_to_apint else "APInt"
    elif isinstance(typ, IndexType):
        return "int"
    assert False and "unsupported type"


CPP_CLASS_KEY = "CPPCLASS"
INDUCTION_KEY = "induction"
OPERATION_NO = "operationNo"


def lowerInductionOps(inductionOp: list[FuncOp]):
    if len(inductionOp) > 0:
        functionSignature = """
{returnedType} {funcName}(ArrayRef<{returnedType}> operands){{
    {returnedType} result={funcName}(operands[0], operands[1]);
    for(int i=2;i<operands.size();++i){{
        result={funcName}(result, operands[i]);
    }}
    return result;
}}

"""
        result = ""
        for func in inductionOp:
            returnedType = func.function_type.outputs.data[0]
            funcName = func.sym_name.data
            returnedType = lowerType(returnedType)
            result += functionSignature.format(
                returnedType=returnedType, funcName=funcName
            )
        return result


def lowerDispatcher(needDispatch: list[FuncOp], is_forward: bool):
    if len(needDispatch) > 0:
        returnedType = needDispatch[0].function_type.outputs.data[0]
        for func in needDispatch:
            if func.function_type.outputs.data[0] != returnedType:
                print(func)
                print(func.function_type.outputs.data[0])
                assert (
                    "we assume all transfer functions have the same returned type"
                    and False
                )
        returnedType = lowerType(returnedType)
        funcName = "naiveDispatcher"
        # we assume all operands have the same type as expr
        # User should tell the generator all operands
        if is_forward:
            expr = "(Operation* op, std::vector<std::vector<llvm::APInt>> operands)"
        else:
            expr = "(Operation* op, std::vector<std::vector<llvm::APInt>> operands, unsigned operationNo)"
        functionSignature = (
            "std::optional<" + returnedType + "> " + funcName + expr + "{{\n{0}}}\n\n"
        )
        indent = "\t"
        dyn_cast = (
            indent
            + "if(auto castedOp=dyn_cast<{0}>(op);castedOp&&{1}){{\n{2}"
            + indent
            + "}}\n"
        )
        return_inst = indent + indent + "return {0}({1});\n"

        def handleOneTransferFunction(func: FuncOp, operationNo: int) -> str:
            blockStr = ""
            for cppClass in func.attributes[CPP_CLASS_KEY]:
                argStr = ""
                if INDUCTION_KEY in func.attributes:
                    argStr = "operands"
                else:
                    if len(func.args) > 0:
                        argStr = "operands[0]"
                    for i in range(1, len(func.args)):
                        argStr += ", operands[" + str(i) + "]"
                ifBody = return_inst.format(func.sym_name.data, argStr)
                if operationNo == -1:
                    operationNoStr = "true"
                else:
                    operationNoStr = "operationNo == " + str(operationNo)
                blockStr += dyn_cast.format(cppClass.data, operationNoStr, ifBody)
            return blockStr

        funcBody = ""
        for func in needDispatch:
            if is_forward:
                funcBody += handleOneTransferFunction(func)
            else:
                operationNo = func.attributes[OPERATION_NO]
                assert isinstance(operationNo, IntegerAttr)
                funcBody += handleOneTransferFunction(func, operationNo.value.data)
        funcBody += indent + "return {};\n"
        return functionSignature.format(funcBody)


def isFunctionCall(opName):
    return opName[0] == "."


def lowerToNonClassMethod(op: Operation):
    returnedType = lowerType(op.results[0].type, op)
    returnedValue = op.results[0].name_hint
    equals = "="
    expr = "("
    if len(op.operands) > 0:
        expr += op.operands[0].name_hint
    for i in range(1, len(op.operands)):
        expr += "," + op.operands[i].name_hint
    expr += ")"
    return (
        indent
        + returnedType
        + " "
        + returnedValue
        + equals
        + operNameToCpp[op.name]
        + expr
        + ends
    )


def lowerToClassMethod(op: Operation, castOperand=None, castResult=None):
    returnedType = lowerType(op.results[0].type, op)
    if castResult is not None:
        returnedValue = op.results[0].name_hint + "_autocast"
    else:
        returnedValue = op.results[0].name_hint
    equals = "="
    expr = op.operands[0].name_hint + operNameToCpp[op.name] + "("
    if castOperand is not None:
        operands = [castOperand(operand) for operand in op.operands]
    else:
        operands = [operand.name_hint for operand in op.operands]
    if len(operands) > 1:
        expr += operands[1]
    for i in range(2, len(operands)):
        expr += "," + operands[i]
    expr += ")"
    result = indent + returnedType + " " + returnedValue + equals + expr + ends
    if castResult is not None:
        return result + castResult(op)
    return result


@singledispatch
def lowerOperation(op):
    returnedType = lowerType(op.results[0].type, op)
    returnedValue = op.results[0].name_hint
    equals = "="
    operandsName = [oper.name_hint for oper in op.operands]
    if isFunctionCall(operNameToCpp[op.name]):
        expr = operandsName[0] + operNameToCpp[op.name] + "("
        if len(operandsName) > 1:
            expr += operandsName[1]
        for i in range(2, len(operandsName)):
            expr += "," + operandsName[i]
        expr += ")"
    else:
        expr = operandsName[0] + operNameToCpp[op.name] + operandsName[1]
    result = indent + returnedType + " " + returnedValue + equals + expr + ends
    return result


@lowerOperation.register
def _(op: CmpOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = op.results[0].name_hint
    equals = "="
    operandsName = [oper.name_hint for oper in op.operands]
    predicate = op.predicate.value.data
    operName = operNameToCpp[op.name][predicate]
    expr = operandsName[0] + operName + "("
    if len(operandsName) > 1:
        expr += operandsName[1]
    for i in range(2, len(operandsName)):
        expr += "," + operandsName[i]
    expr += ")"
    return indent + returnedType + " " + returnedValue + equals + expr + ends


@lowerOperation.register
def _(op: arith.CmpiOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = op.results[0].name_hint
    equals = "="
    operandsName = [oper.name_hint for oper in op.operands]
    assert len(operandsName) == 2
    predicate = op.predicate.value.data
    operName = operNameToCpp[op.name][predicate]
    expr = "(" + operandsName[0] + operName + operandsName[1]
    expr += ")"
    return indent + returnedType + " " + returnedValue + equals + expr + ends


@lowerOperation.register
def _(op: arith.SelectOp):
    returnedType = lowerType(op.operands[1].type, op)
    returnedValue = op.results[0].name_hint
    equals = "="
    operandsName = [oper.name_hint for oper in op.operands]
    operator = operNameToCpp[op.name]
    expr = ""
    for i in range(len(operandsName)):
        expr += operandsName[i] + " "
        if i < len(operator):
            expr += operator[i] + " "
    return indent + returnedType + " " + returnedValue + equals + expr + ends


@lowerOperation.register
def _(op: SelectOp):
    returnedType = lowerType(op.operands[1].type, op)
    returnedValue = op.results[0].name_hint
    equals = "="
    operandsName = [oper.name_hint for oper in op.operands]
    operator = operNameToCpp[op.name]
    expr = ""
    for i in range(len(operandsName)):
        expr += operandsName[i] + " "
        if i < len(operator):
            expr += operator[i] + " "
    return indent + returnedType + " " + returnedValue + equals + expr + ends


@lowerOperation.register
def _(op: GetOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = op.results[0].name_hint
    equals = "="
    index = op.attributes["index"].value.data
    return (
        indent
        + returnedType
        + " "
        + returnedValue
        + equals
        + op.operands[0].name_hint
        + operNameToCpp[op.name].format(index)
        + ends
    )


@lowerOperation.register
def _(op: MakeOp):
    returnedType = lowerType(op.results[0].type, op)
    returnedValue = op.results[0].name_hint
    equals = "="
    expr = ""
    if len(op.operands) > 0:
        expr += op.operands[0].name_hint
    for i in range(1, len(op.operands)):
        expr += "," + op.operands[i].name_hint
    return (
        indent
        + returnedType
        + " "
        + returnedValue
        + equals
        + returnedType
        + operNameToCpp[op.name].format(expr)
        + ends
    )


@lowerOperation.register
def _(op: UMulOverflowOp):
    varDecls = "bool " + op.results[0].name_hint + ends
    expr = op.operands[0].name_hint + operNameToCpp[op.name] + "("
    expr += op.operands[1].name_hint + "," + op.results[0].name_hint
    expr += ")"
    result = varDecls + "\t" + expr + ends
    return indent + result


@lowerOperation.register
def _(op: NegOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = op.results[0].name_hint
    equals = "="
    return (
        indent
        + returnedType
        + " "
        + returnedValue
        + equals
        + operNameToCpp[op.name]
        + op.operands[0].name_hint
        + ends
    )


@lowerOperation.register
def _(op: ReturnOp):
    opName = operNameToCpp[op.name] + " "
    operand = op.arguments[0].name_hint
    return indent + opName + operand + ends


"""
@lowerOperation.register
def _(op: FromArithOp):
    opTy = op.op.type
    assert isinstance(opTy, IntegerType)
    size = opTy.width.data
    returnedType = "APInt"
    returnedValue = op.results[0].name_hint
    return (
        indent
        + returnedType
        + " "
        + returnedValue
        + "("
        + str(size)
        + ", "
        + op.op.name_hint
        + ")"
        + ends
    )
"""


@lowerOperation.register
def _(op: arith.ConstantOp):
    value = op.value.value.data
    assert isinstance(op.results[0].type, IntegerType)
    size = op.results[0].type.width.data
    returnedType = "int"
    if value > ((1 << 31) - 1):
        assert False and "arith constant overflow maximal int"
    returnedValue = op.results[0].name_hint
    return indent + returnedType + " " + returnedValue + " = " + str(value) + ends


@lowerOperation.register
def _(op: Constant):
    value = op.value.value.data
    returnedType = lowerType(op.results[0].type)
    returnedValue = op.results[0].name_hint
    return (
        indent
        + returnedType
        + " "
        + returnedValue
        + "("
        + op.operands[0].name_hint
        + ".getBitWidth(),"
        + str(value)
        + ")"
        + ends
    )


@lowerOperation.register
def _(op: GetAllOnesOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = op.results[0].name_hint
    opName = operNameToCpp[op.name]
    return (
        indent
        + returnedType
        + " "
        + returnedValue
        + " = "
        + opName
        + "("
        + op.operands[0].name_hint
        + ".getBitWidth()"
        + ")"
        + ends
    )


@lowerOperation.register
def _(op: CallOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = op.results[0].name_hint
    callee = op.callee.string_value() + "("
    operandsName = [oper.name_hint for oper in op.operands]
    expr = ""
    if len(operandsName) > 0:
        expr += operandsName[0]
    for i in range(1, len(operandsName)):
        expr += "," + operandsName[i]
    expr += ")"
    return indent + returnedType + " " + returnedValue + "=" + callee + expr + ends


@lowerOperation.register
def _(op: FuncOp):
    def lowerArgs(arg):
        return lowerType(arg.type) + " " + arg.name_hint

    returnedType = lowerType(op.function_type.outputs.data[0])
    funcName = op.sym_name.data
    expr = "("
    if len(op.args) > 0:
        expr += lowerArgs(op.args[0])
    for i in range(1, len(op.args)):
        expr += "," + lowerArgs(op.args[i])
    expr += ")"
    # return returnedType + " " + funcName + expr + "{{\n{0}}}\n\n"
    return returnedType + " " + funcName + expr + "{\n"


def castToAPIntFromUnsigned(op: Operation):
    lastReturn = op.results[0].name_hint + "_autocast"
    apInt = None
    for operand in op.operands:
        if isinstance(operand.type, TransIntegerType):
            apInt = operand.name_hint
            break
    returnedType = "APInt"
    returnedValue = op.results[0].name_hint
    return (
        indent
        + returnedType
        + " "
        + returnedValue
        + "("
        + apInt
        + ".getBitWidth(),"
        + lastReturn
        + ")"
        + ends
    )


@lowerOperation.register
def _(op: IntersectsOp):
    return lowerToClassMethod(op, None, None)


@lowerOperation.register
def _(op: CountLOneOp):
    return lowerToClassMethod(op, None, castToAPIntFromUnsigned)


@lowerOperation.register
def _(op: CountLZeroOp):
    return lowerToClassMethod(op, None, castToAPIntFromUnsigned)


@lowerOperation.register
def _(op: CountROneOp):
    return lowerToClassMethod(op, None, castToAPIntFromUnsigned)


@lowerOperation.register
def _(op: CountRZeroOp):
    return lowerToClassMethod(op, None, castToAPIntFromUnsigned)


def castToUnisgnedFromAPInt(operand):
    if isinstance(operand.type, TransIntegerType):
        return operand.name_hint + ".getZExtValue()"
    return operand.name_hint


@lowerOperation.register
def _(op: SetHighBitsOp):
    returnedType = lowerType(op.results[0].type, op)
    returnedValue = op.results[0].name_hint
    equals = "=" + op.operands[0].name_hint + ends + "\t"
    expr = op.results[0].name_hint + operNameToCpp[op.name] + "("
    operands = op.operands[1].name_hint + ".getZExtValue()"
    expr = expr + operands + ")"
    result = returnedType + " " + returnedValue + equals + expr + ends
    return indent + result


@lowerOperation.register
def _(op: SetLowBitsOp):
    returnedType = lowerType(op.results[0].type, op)
    returnedValue = op.results[0].name_hint
    equals = "=" + op.operands[0].name_hint + ends + "\t"
    expr = op.results[0].name_hint + operNameToCpp[op.name] + "("
    operands = op.operands[1].name_hint + ".getZExtValue()"
    expr = expr + operands + ")"
    result = returnedType + " " + returnedValue + equals + expr + ends
    return indent + result


@lowerOperation.register
def _(op: GetLowBitsOp):
    return lowerToClassMethod(op, castToUnisgnedFromAPInt)


@lowerOperation.register
def _(op: GetBitWidthOp):
    return lowerToClassMethod(op, None, castToAPIntFromUnsigned)


# op1 < op2? op1: op2
@lowerOperation.register
def _(op: SMaxOp):
    returnedType = lowerType(op.operands[0].type, op)
    returnedValue = op.results[0].name_hint
    operands = [operand.name_hint for operand in op.operands]
    operator = operNameToCpp[op.name]
    equals = "="
    expr = (
        operands[0]
        + operator[0]
        + "("
        + operands[1]
        + ")"
        + operator[1]
        + operands[0]
        + operator[2]
        + operands[1]
    )
    result = returnedType + " " + returnedValue + equals + expr + ends
    return indent + result


@lowerOperation.register
def _(op: SMinOp):
    returnedType = lowerType(op.operands[0].type, op)
    returnedValue = op.results[0].name_hint
    operands = [operand.name_hint for operand in op.operands]
    operator = operNameToCpp[op.name]
    equals = "="
    expr = (
        operands[0]
        + operator[0]
        + "("
        + operands[1]
        + ")"
        + operator[1]
        + operands[0]
        + operator[2]
        + operands[1]
    )
    result = returnedType + " " + returnedValue + equals + expr + ends
    return indent + result


@lowerOperation.register
def _(op: UMaxOp):
    returnedType = lowerType(op.operands[0].type, op)
    returnedValue = op.results[0].name_hint
    operands = [operand.name_hint for operand in op.operands]
    operator = operNameToCpp[op.name]
    equals = "="
    expr = (
        operands[0]
        + operator[0]
        + "("
        + operands[1]
        + ")"
        + operator[1]
        + operands[0]
        + operator[2]
        + operands[1]
    )
    result = returnedType + " " + returnedValue + equals + expr + ends
    return indent + result


@lowerOperation.register
def _(op: UMinOp):
    returnedType = lowerType(op.operands[0].type, op)
    returnedValue = op.results[0].name_hint
    operands = [operand.name_hint for operand in op.operands]
    operator = operNameToCpp[op.name]
    equals = "="
    expr = (
        operands[0]
        + operator[0]
        + "("
        + operands[1]
        + ")"
        + operator[1]
        + operands[0]
        + operator[2]
        + operands[1]
    )
    result = returnedType + " " + returnedValue + equals + expr + ends
    return indent + result


@lowerOperation.register
def _(op: ShlOp):
    return lowerToClassMethod(op, castToUnisgnedFromAPInt)


@lowerOperation.register
def _(op: AShrOp):
    return lowerToClassMethod(op, castToUnisgnedFromAPInt)


@lowerOperation.register
def _(op: LShrOp):
    return lowerToClassMethod(op, castToUnisgnedFromAPInt)


@lowerOperation.register
def _(op: ExtractOp):
    return lowerToClassMethod(op, castToUnisgnedFromAPInt)


@lowerOperation.register
def _(op: ConcatOp):
    return lowerToClassMethod(op)


@lowerOperation.register
def _(op: ConstRangeForOp):
    loopBody = op.body.block
    lowerBound = op.lb.name_hint
    upperBound = op.ub.name_hint
    step = op.step.name_hint

    indvar, *block_iter_args = loopBody.args
    iter_args = op.iter_args

    global indent
    loopBefore = ""
    for i, blk_arg in enumerate(block_iter_args):
        iter_type = lowerType(iter_args[i].type, iter_args[i].owner)
        iter_name = blk_arg.name_hint
        loopBefore += (
            indent + iter_type + " " + iter_name + " = " + iter_args[i].name_hint + ends
        )

    loopFor = indent + "for(APInt {0} = {1}; {0}.ule({2}); {0}+={3}){{\n".format(
        indvar.name_hint, lowerBound, upperBound, step
    )
    indent += "\t"
    """
    mainLoop=""
    for loopOp in loopBody.ops:
        mainLoop+=(indent  + indent+ lowerOperation(loopOp))
    endLoopFor=indent+"}\n"
    """
    return loopBefore + loopFor


@lowerOperation.register
def _(op: NextLoopOp):
    loopBlock = op.parent_block()
    indvar, *block_iter_args = loopBlock.args
    global indent
    assignments = ""
    for i, arg in enumerate(op.operands):
        assignments += (
            indent + block_iter_args[i].name_hint + " = " + arg.name_hint + ends
        )
    indent = indent[:-1]
    endLoopFor = indent + "}\n"
    loopOp = loopBlock.parent_op()
    for i, res in enumerate(loopOp.results):
        endLoopFor += (
            indent
            + lowerType(res.type, loopOp)
            + " "
            + res.name_hint
            + " = "
            + block_iter_args[i].name_hint
            + ends
        )
    return assignments + endLoopFor


@lowerOperation.register
def _(op: RepeatOp):
    returnedType = lowerType(op.operands[0].type, op)
    returnedValue = op.results[0].name_hint
    arg0_name = op.operands[0].name_hint
    count = op.operands[1].name_hint
    initExpr = indent + returnedType + " " + returnedValue + " = " + arg0_name + ends
    forHead = (
        indent
        + "for(APInt i("
        + count
        + ".getBitWidth(),1);i.ult("
        + count
        + ");++i){\n"
    )
    forBody = (
        indent
        + "\t"
        + returnedValue
        + " = "
        + returnedValue
        + ".concat("
        + arg0_name
        + ")"
        + ends
    )
    forEnd = indent + "}\n"
    return initExpr + forHead + forBody + forEnd


@lowerOperation.register
def _(op: AddPoisonOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = op.results[0].name_hint
    opName = operNameToCpp[op.name]
    return (
        indent
        + returnedType
        + " "
        + returnedValue
        + " = "
        + op.operands[0].name_hint
        + ends
    )


@lowerOperation.register
def _(op: RemovePoisonOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = op.results[0].name_hint
    opName = operNameToCpp[op.name]
    return (
        indent
        + returnedType
        + " "
        + returnedValue
        + " = "
        + op.operands[0].name_hint
        + ends
    )
