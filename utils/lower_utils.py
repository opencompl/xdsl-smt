from dialects.transfer import (
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
    GetLowBitsOp,
    GetBitWidthOp,
    UMulOverflowOp,
    SMinOp,
    SMaxOp,
    UMinOp,
    UMaxOp,
    TransIntegerType,
)
from xdsl.dialects.func import FuncOp, Return, Call
from xdsl.pattern_rewriter import *
from functools import singledispatch
from typing import TypeVar, cast
from xdsl.dialects.builtin import Signedness, IntegerType, IndexType
from xdsl.ir import Operation
from xdsl.dialects.arith import (
    AndI,
    Select,
)

operNameToCpp = {
    "transfer.and": "&",
    "arith.andi": "&",
    "transfer.add": "+",
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
        "uge",
    ],
    "transfer.make": "std::make_tuple",
    "transfer.get": "std::get<{0}>",
    "transfer.umin": [".ule", "?", ":"],
    "transfer.smin": [".sle", "?", ":"],
    "transfer.umax": [".ugt", "?", ":"],
    "transfer.smax": [".sgt", "?", ":"],
    "func.return": "return",
    "transfer.constant": "APInt",
    "arith.select": ["?", ":"],
}

unsignedReturnedType = {
    CountLOneOp,
    CountLZeroOp,
    CountROneOp,
    CountRZeroOp,
    GetBitWidthOp,
}

ends = ";\n"


def lowerType(typ, specialOp=None):
    if specialOp is not None:
        for op in unsignedReturnedType:
            if isinstance(specialOp, op):
                return "unsigned"
    if isinstance(typ, TransIntegerType):
        return "APInt"
    elif isinstance(typ, AbstractValueType):
        typeName = "std::tuple<"
        fields = typ.get_fields()
        typeName += lowerType(fields[0])
        for i in range(1, len(fields)):
            typeName += ","
            typeName += lowerType(fields[i])
        typeName += ">"
        return typeName
    elif isinstance(typ, IntegerType):
        return "int"
    elif isinstance(typ, IndexType):
        return "int"
    print(typ)
    assert False and "unsupported type"


CPP_CLASS_KEY = "CPPCLASS"


def lowerDispatcher(needDispatch: list[FuncOp]):
    if len(needDispatch) > 0:
        returnedType = needDispatch[0].function_type.outputs.data[0]
        for func in needDispatch:
            if func.function_type.outputs.data[0] != returnedType:
                assert (
                    "we assume all transfer functions have the same returned type"
                    and False
                )
        returnedType = lowerType(returnedType)
        funcName = "naiveDispatcher"
        # we assume all operands have the same type as expr
        expr = "(Operation* op, ArrayRef<" + returnedType + "> operands)"
        functionSignature = (
            "std::optional<" + returnedType + "> " + funcName + expr + "{{\n{0}}}\n\n"
        )
        indent = "\t"
        dyn_cast = (
            indent
            + "if(auto castedOp=dyn_cast<{0}>(op);castedOp){{\n{1}"
            + indent
            + "}}\n"
        )
        return_inst = indent + indent + "return {0}({1});\n"

        def handleOneTransferFunction(func: FuncOp) -> str:
            blockStr = ""
            for cppClass in func.attributes[CPP_CLASS_KEY]:
                argStr = ""
                if len(func.args) > 0:
                    argStr = "operands[0]"
                for i in range(1, len(func.args)):
                    argStr += ", operands[" + str(i) + "]"
                ifBody = return_inst.format(func.sym_name.data, argStr)
                blockStr += dyn_cast.format(cppClass.data, ifBody)
            return blockStr

        funcBody = ""
        for func in needDispatch:
            funcBody += handleOneTransferFunction(func)
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
        returnedType
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
    result = returnedType + " " + returnedValue + equals + expr + ends
    if castResult is not None:
        return result + "\t" + castResult(op)
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
    return returnedType + " " + returnedValue + equals + expr + ends


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
    return returnedType + " " + returnedValue + equals + expr + ends


@lowerOperation.register
def _(op: Select):
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
    return returnedType + " " + returnedValue + equals + expr + ends


@lowerOperation.register
def _(op: GetOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = op.results[0].name_hint
    equals = "="
    index = op.attributes["index"].value.data
    return (
        returnedType
        + " "
        + returnedValue
        + equals
        + operNameToCpp[op.name].format(index)
        + "("
        + op.operands[0].name_hint
        + ")"
        + ends
    )


@lowerOperation.register
def _(op: MakeOp):
    return lowerToNonClassMethod(op)


@lowerOperation.register
def _(op: UMulOverflowOp):
    varDecls = "bool " + op.results[0].name_hint + ends
    expr = op.operands[0].name_hint + operNameToCpp[op.name] + "("
    expr += op.operands[1].name_hint + "," + op.results[0].name_hint
    expr += ")"
    result = varDecls + "\t" + expr + ends
    return result


@lowerOperation.register
def _(op: NegOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = op.results[0].name_hint
    equals = "="
    return (
        returnedType
        + " "
        + returnedValue
        + equals
        + operNameToCpp[op.name]
        + op.operands[0].name_hint
        + ends
    )


@lowerOperation.register
def _(op: Return):
    opName = operNameToCpp[op.name] + " "
    operand = op.arguments[0].name_hint
    return opName + operand + ends


@lowerOperation.register
def _(op: Constant):
    value = op.value.value.data
    returnedType = lowerType(op.results[0].type)
    returnedValue = op.results[0].name_hint
    return (
        returnedType
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
def _(op: Call):
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
    return returnedType + " " + returnedValue + "=" + callee + expr + ends


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
    return returnedType + " " + funcName + expr + "{{\n{0}}}\n\n"


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
        returnedType
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
    return result


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
    return result


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
    return result


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
    return result


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
    return result
