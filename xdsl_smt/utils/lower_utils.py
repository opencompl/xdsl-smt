from functools import singledispatch
from typing import Callable

import xdsl.dialects.arith as arith
from xdsl.dialects.builtin import IndexType, IntegerType
from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
from xdsl.ir import Attribute, Block, BlockArgument, Operation, SSAValue

from ..dialects.transfer import (
    AbstractValueType,
    AddPoisonOp,
    AShrOp,
    ClearHighBitsOp,
    ClearLowBitsOp,
    ClearSignBitOp,
    CmpOp,
    ConcatOp,
    Constant,
    ConstRangeForOp,
    CountLOneOp,
    CountLZeroOp,
    CountROneOp,
    CountRZeroOp,
    ExtractOp,
    GetAllOnesOp,
    GetBitWidthOp,
    GetHighBitsOp,
    GetLowBitsOp,
    GetOp,
    GetSignedMaxValueOp,
    GetSignedMinValueOp,
    IntersectsOp,
    LShrOp,
    MakeOp,
    NegOp,
    NextLoopOp,
    RemovePoisonOp,
    RepeatOp,
    SAddOverflowOp,
    SDivOp,
    SelectOp,
    SetHighBitsOp,
    SetLowBitsOp,
    SetSignBitOp,
    ShlOp,
    SMaxOp,
    SMinOp,
    SMulOverflowOp,
    SRemOp,
    SShlOverflowOp,
    TransIntegerType,
    TupleType,
    UAddOverflowOp,
    UDivOp,
    UMaxOp,
    UMinOp,
    UMulOverflowOp,
    URemOp,
    UShlOverflowOp,
)

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
    "transfer.udiv": ".udiv",
    "transfer.sdiv": ".sdiv",
    "transfer.urem": ".urem",
    "transfer.srem": ".srem",
    "transfer.umul_overflow": ".umul_ov",
    "transfer.smul_overflow": ".smul_ov",
    "transfer.uadd_overflow": ".uadd_ov",
    "transfer.sadd_overflow": ".sadd_ov",
    "transfer.ushl_overflow": ".ushl_ov",
    "transfer.sshl_overflow": ".sshl_ov",
    "transfer.get_bit_width": ".getBitWidth",
    "transfer.countl_zero": ".countl_zero",
    "transfer.countr_zero": ".countr_zero",
    "transfer.countl_one": ".countl_one",
    "transfer.countr_one": ".countr_one",
    "transfer.get_high_bits": ".getHiBits",
    "transfer.get_low_bits": ".getLoBits",
    "transfer.set_high_bits": ".setHighBits",
    "transfer.set_low_bits": ".setLowBits",
    "transfer.clear_high_bits": ".clearHighBits",
    "transfer.clear_low_bits": ".clearLowBits",
    "transfer.set_sign_bit": ".setSignBit",
    "transfer.clear_sign_bit": ".clearSignBit",
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
    "transfer.make": "{{{0}}}",
    "transfer.get": "[{0}]",
    "transfer.shl": ".shl",
    "transfer.ashr": ".ashr",
    "transfer.lshr": ".lshr",
    "transfer.concat": ".concat",
    "transfer.extract": ".extractBits",
    "transfer.umin": "A::APIntOps::umin",
    "transfer.smin": "A::APIntOps::smin",
    "transfer.umax": "A::APIntOps::umax",
    "transfer.smax": "A::APIntOps::smax",
    "func.return": "return",
    "transfer.constant": "APInt",
    "arith.select": ["?", ":"],
    "arith.cmpi": ["==", "!=", "<", "<=", ">", ">="],
    "transfer.get_all_ones": "APInt::getAllOnes",
    "transfer.get_signed_max_value": "APInt::getSignedMaxValue",
    "transfer.get_signed_min_value": "APInt::getSignedMinValue",
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
    "comb.shrs": ".ashr",
    "comb.shru": ".lshr",
    "comb.shl": ".shl",
    "comb.extract": ".extractBits",
    "comb.concat": ".concat",
}
# transfer.constRangeLoop and NextLoop are controller operations, should be handle specially

# consts
EQ = " = "
END = ";\n"
IDNT = "\t"

VAL_EXCEEDS_BW = "{1}.uge({1}.getBitWidth())"
RHS_IS_ZERO = "{1} == 0"
RET_ZERO = "{0} = APInt({1}.getBitWidth(), 0)"
RET_ONE = "{0} = APInt({1}.getBitWidth(), 1)"
RET_ONES = "{0} = APInt({1}.getBitWidth(), -1)"
RET_SIGN_MIN_VAL = "{0} = APInt::getSignedMinValue({1}.getBitWidth())"
RET_LHS = "{0} = {1}"

SHIFT_ACTION = (VAL_EXCEEDS_BW, RET_ZERO)
ASHR_ACTION0 = VAL_EXCEEDS_BW + " && {0}.isSignBitSet()", RET_ONES
ASHR_ACTION1 = VAL_EXCEEDS_BW + " && {0}.isSignBitClear()", RET_ZERO
REM_ACTION = RHS_IS_ZERO, RET_LHS
DIV_ACTION = RHS_IS_ZERO, RET_ONES
SDIV_ACTION0 = ("{0}.isMinSignedValue() && {1} == -1", RET_SIGN_MIN_VAL)
SDIV_ACTION1 = (RHS_IS_ZERO + " && {0}.isNonNegative()", RET_ONES)
SDIV_ACTION2 = (RHS_IS_ZERO + " && {0}.isNegative()", RET_ONE)

op_to_cons: dict[type[Operation], list[tuple[str, str]]] = {
    ShlOp: [SHIFT_ACTION],
    LShrOp: [SHIFT_ACTION],
    UDivOp: [DIV_ACTION],
    URemOp: [REM_ACTION],
    SRemOp: [REM_ACTION],
    AShrOp: [ASHR_ACTION0, ASHR_ACTION1],
    SDivOp: [SDIV_ACTION0, SDIV_ACTION1, SDIV_ACTION2],
}

# lowering config
use_apint = False
use_custom_vec = False
use_llvm_kb = False


def set_use_apint(f: bool) -> None:
    global use_apint
    use_apint = f


def set_use_custom_vec(f: bool) -> None:
    global use_custom_vec
    use_custom_vec = f


def set_use_llvm_kb(f: bool) -> None:
    global use_llvm_kb
    use_llvm_kb = f


# helpers
def get_ret_val(op: Operation) -> str:
    ret_val = op.results[0].name_hint
    assert ret_val
    return ret_val


def get_op_names(op: Operation) -> list[str]:
    return [oper.name_hint for oper in op.operands if oper.name_hint]


def get_operand(op: Operation, idx: int) -> str:
    name = op.operands[idx].name_hint
    assert name
    return name


def get_op_str(op: Operation) -> str:
    op_name = operNameToCpp[op.name]
    assert isinstance(op_name, str)
    return op_name


def lowerType(typ: Attribute, specialOp: Operation | Block | None = None) -> str:
    unsigned_ret_type = {
        CountLOneOp,
        CountLZeroOp,
        CountROneOp,
        CountRZeroOp,
        GetBitWidthOp,
    }

    if specialOp is not None:
        for op in unsigned_ret_type:
            if isinstance(specialOp, op):
                return "unsigned"

    if isinstance(typ, TransIntegerType) or (
        isinstance(typ, IntegerType) and use_apint
    ):
        return "const APInt"
    elif isinstance(typ, AbstractValueType) or isinstance(typ, TupleType):
        fields = typ.get_fields()
        typeName = lowerType(fields[0])
        for i in range(1, len(fields)):
            assert lowerType(fields[i]) == typeName

        if use_custom_vec:
            return "Vec<" + str(len(fields)) + ">"
        elif use_llvm_kb:
            assert len(fields) == 2
            return "const llvm::KnownBits"
        else:
            return "std::vector<" + typeName + ">"
    elif isinstance(typ, IndexType) or isinstance(typ, IntegerType):
        return "int"

    raise ValueError(f"unsupported type: {type(typ)}")


def isFunctionCall(opName: str) -> bool:
    return opName[0] == "."


def lowerToClassMethod(
    op: Operation,
    castOperand: Callable[[SSAValue | str], str] | None = None,
    castResult: Callable[[Operation], str] | None = None,
) -> str:
    ret_ty = lowerType(op.results[0].type, op)
    ret_val = get_ret_val(op)

    if castResult is not None:
        ret_val += "_autocast"
    expr = get_operand(op, 0) + get_op_str(op) + "("

    if castOperand is not None:
        operands = [castOperand(operand) for operand in op.operands]
    else:
        operands = get_op_names(op)

    if len(operands) > 1:
        expr += operands[1]
    for i in range(2, len(operands)):
        expr += "," + operands[i]

    expr += ")"

    if type(op) in op_to_cons:
        conds, actions = zip(*op_to_cons[type(op)])  # type: ignore

        og_op_names = get_op_names(op)
        conds: list[str] = [cond.format(*og_op_names) for cond in conds]
        actions: list[str] = [act.format(ret_val, *og_op_names) for act in actions]

        if_fmt = "if ({cond}) {{\n" + IDNT + IDNT + "{act}" + END + IDNT + "}}"

        ifs = " else ".join(
            [if_fmt.format(cond=c, act=a) for c, a in zip(conds, actions)]
        )

        final_else_br = IDNT + IDNT + ret_val + EQ + expr + END

        result = IDNT + ret_ty + " " + ret_val + END
        result += IDNT + ifs + " else {\n" + final_else_br + IDNT + "}\n"

    else:
        result = IDNT + ret_ty + " " + ret_val + EQ + expr + END

    if castResult is not None:
        return result + castResult(op)

    return result


@singledispatch
def lowerOperation(op: Operation) -> str:
    returnedType = lowerType(op.results[0].type, op)
    returnedValue = get_ret_val(op)
    operandsName = get_op_names(op)
    op_str = get_op_str(op)

    if isFunctionCall(op_str):
        expr = operandsName[0] + op_str + "("
        if len(operandsName) > 1:
            expr += operandsName[1]
        for i in range(2, len(operandsName)):
            expr += "," + operandsName[i]
        expr += ")"
    else:
        expr = operandsName[0] + op_str + operandsName[1]

    return IDNT + returnedType + " " + returnedValue + EQ + expr + END


@lowerOperation.register
def _(op: CmpOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = get_ret_val(op)
    operandsName = get_op_names(op)
    predicate = op.predicate.value.data
    operName = operNameToCpp[op.name][predicate]
    expr = operandsName[0] + operName + "("
    if len(operandsName) > 1:
        expr += operandsName[1]
    for i in range(2, len(operandsName)):
        expr += "," + operandsName[i]
    expr += ")"

    return IDNT + returnedType + " " + returnedValue + EQ + expr + END


@lowerOperation.register
def _(op: arith.CmpiOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = get_ret_val(op)
    operandsName = get_op_names(op)
    assert len(operandsName) == 2
    predicate = op.predicate.value.data
    operName = operNameToCpp[op.name][predicate]
    expr = "(" + operandsName[0] + operName + operandsName[1] + ")"

    return IDNT + returnedType + " " + returnedValue + EQ + expr + END


@lowerOperation.register
def _(op: arith.SelectOp):
    returnedType = lowerType(op.operands[1].type, op)
    returnedValue = get_ret_val(op)
    operandsName = get_op_names(op)
    operator = operNameToCpp[op.name]
    expr = ""
    for i in range(len(operandsName)):
        expr += operandsName[i] + " "
        if i < len(operator):
            expr += operator[i] + " "

    return IDNT + returnedType + " " + returnedValue + EQ + expr + END


@lowerOperation.register
def _(op: SelectOp):
    returnedType = lowerType(op.operands[1].type, op)
    returnedValue = get_ret_val(op)
    operandsName = get_op_names(op)
    operator = operNameToCpp[op.name]
    expr = ""
    for i in range(len(operandsName)):
        expr += operandsName[i] + " "
        if i < len(operator):
            expr += operator[i] + " "

    return IDNT + returnedType + " " + returnedValue + EQ + expr + END


@lowerOperation.register
def _(op: GetOp) -> str:
    returnedType = lowerType(op.results[0].type)
    returnedValue = get_ret_val(op)
    index = op.attributes["index"].value.data  # type: ignore

    if use_llvm_kb:
        return (
            IDNT
            + returnedType
            + " "
            + returnedValue
            + EQ
            + get_operand(op, 0)
            + (".Zero" if index == 0 else ".One")
            + END
        )

    else:
        return (
            IDNT
            + returnedType
            + " "
            + returnedValue
            + EQ
            + get_operand(op, 0)
            + get_op_str(op).format(index)  # type: ignore
            + END
        )


@lowerOperation.register
def _(op: MakeOp) -> str:
    returnedValue = get_ret_val(op)

    if use_llvm_kb and isinstance(op.results[0].type, AbstractValueType):
        s = f"{IDNT}llvm::KnownBits {returnedValue}{END}"
        s += f"{IDNT}{returnedValue}.Zero = {get_operand(op, 0)}{END}"
        s += f"{IDNT}{returnedValue}.One = {get_operand(op, 1)}{END}"
        return s

    returnedType = lowerType(op.results[0].type, op)
    expr = ""
    if len(op.operands) > 0:
        expr += get_operand(op, 0)
    for i in range(1, len(op.operands)):
        expr += "," + get_operand(op, i)

    return (
        IDNT
        + returnedType
        + " "
        + returnedValue
        + EQ
        + returnedType
        + get_op_str(op).format(expr)
        + END
    )


def trivial_overflow_predicate(op: Operation) -> str:
    returnedValue = get_ret_val(op)
    varDecls = "bool " + returnedValue + END
    expr = get_operand(op, 0) + get_op_str(op) + "("
    expr += get_operand(op, 1) + "," + returnedValue + ")"
    result = varDecls + IDNT + expr + END
    return IDNT + result


@lowerOperation.register
def _(op: UMulOverflowOp):
    return trivial_overflow_predicate(op)


@lowerOperation.register
def _(op: SMulOverflowOp):
    return trivial_overflow_predicate(op)


@lowerOperation.register
def _(op: UAddOverflowOp):
    return trivial_overflow_predicate(op)


@lowerOperation.register
def _(op: SAddOverflowOp):
    return trivial_overflow_predicate(op)


@lowerOperation.register
def _(op: SShlOverflowOp):
    return trivial_overflow_predicate(op)


@lowerOperation.register
def _(op: UShlOverflowOp):
    return trivial_overflow_predicate(op)


@lowerOperation.register
def _(op: NegOp) -> str:
    ret_type = lowerType(op.results[0].type)
    ret_val = get_ret_val(op)
    op_str = get_op_str(op)
    operand = get_operand(op, 0)

    return IDNT + ret_type + " " + ret_val + EQ + op_str + operand + END


@lowerOperation.register
def _(op: ReturnOp) -> str:
    opName = get_op_str(op) + " "
    operand = op.arguments[0].name_hint
    assert operand

    return IDNT + opName + operand + END


@lowerOperation.register
def _(op: arith.ConstantOp):
    value = op.value.value.data  # type: ignore
    assert isinstance(value, int) or isinstance(value, float)
    assert isinstance(op.results[0].type, IntegerType)
    size = op.results[0].type.width.data
    max_val_plus_one = 1 << size
    returnedType = "int"
    if value >= (1 << 31):
        assert False and "arith constant overflow maximal int"
    returnedValue = get_ret_val(op)
    return (
        IDNT
        + returnedType
        + " "
        + returnedValue
        + EQ
        + str((value + max_val_plus_one) % max_val_plus_one)
        + END
    )


@lowerOperation.register
def _(op: Constant):
    value = op.value.value.data
    returnedType = lowerType(op.results[0].type)
    returnedValue = get_ret_val(op)
    return (
        IDNT
        + returnedType
        + " "
        + returnedValue
        + "("
        + get_operand(op, 0)
        + ".getBitWidth(),"
        + str(value)
        + ")"
        + END
    )


@lowerOperation.register
def _(op: GetAllOnesOp):
    ret_type = lowerType(op.results[0].type)
    ret_val = get_ret_val(op)
    op_name = get_op_str(op)

    return (
        IDNT
        + ret_type
        + " "
        + ret_val
        + EQ
        + op_name
        + "("
        + get_operand(op, 0)
        + ".getBitWidth()"
        + ")"
        + END
    )


@lowerOperation.register
def _(op: GetSignedMaxValueOp):
    ret_type = lowerType(op.results[0].type)
    ret_val = get_ret_val(op)
    op_name = get_op_str(op)

    return (
        IDNT
        + ret_type
        + " "
        + ret_val
        + EQ
        + op_name
        + "("
        + get_operand(op, 0)
        + ".getBitWidth()"
        + ")"
        + END
    )


@lowerOperation.register
def _(op: GetSignedMinValueOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = get_ret_val(op)
    op_name = get_op_str(op)

    return (
        IDNT
        + returnedType
        + " "
        + returnedValue
        + EQ
        + op_name
        + "("
        + get_operand(op, 0)
        + ".getBitWidth()"
        + ")"
        + END
    )


@lowerOperation.register
def _(op: CallOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = get_ret_val(op)
    callee = op.callee.string_value() + "("
    operandsName = get_op_names(op)
    expr = ""
    if len(operandsName) > 0:
        expr += operandsName[0]
    for i in range(1, len(operandsName)):
        expr += "," + operandsName[i]
    expr += ")"
    return IDNT + returnedType + " " + returnedValue + EQ + callee + expr + END


def set_clear_bits(
    op: SetHighBitsOp | SetLowBitsOp | ClearHighBitsOp | ClearLowBitsOp,
) -> str:
    ret_ty = lowerType(op.results[0].type, op)
    ret_val = get_ret_val(op)
    arg = get_operand(op, 0)
    count = get_operand(op, 1)
    op_str = get_op_str(op)

    set_val = f"{IDNT}{ret_ty} {ret_val} = {arg};\n"
    cond = f"{count}.ule({count}.getBitWidth())"
    if_br = f"{IDNT}{IDNT}{ret_val}{op_str}({count}.getZExtValue());\n"
    el_br = f"{IDNT}{IDNT}{ret_val}{op_str}({count}.getBitWidth());\n"

    return f"{set_val}{IDNT}if ({cond})\n{if_br}{IDNT}else\n{el_br}"


@lowerOperation.register
def _(op: FuncOp):
    def lowerArgs(arg: BlockArgument) -> str:
        global use_apint
        assert arg.name_hint
        s = f"{lowerType(arg.type)} {arg.name_hint}"
        if (
            isinstance(arg.type, AbstractValueType)
            or isinstance(arg.type, TupleType)
            or isinstance(arg.type, TransIntegerType)
            or (isinstance(arg.type, IntegerType) and use_apint)
        ):
            s = f"{lowerType(arg.type)} &{arg.name_hint}"

        return s

    returnedType = lowerType(op.function_type.outputs.data[0])
    funcName = op.sym_name.data
    expr = "("
    if len(op.args) > 0:
        expr += lowerArgs(op.args[0])
    for i in range(1, len(op.args)):
        expr += "," + lowerArgs(op.args[i])
    expr += ")"

    return returnedType + " " + funcName + expr + "{\n"  # }


def castToAPIntFromUnsigned(op: Operation) -> str:
    returnedValue = get_ret_val(op)
    lastReturn = returnedValue + "_autocast"
    apInt = None
    for operand in op.operands:
        if isinstance(operand.type, TransIntegerType):
            apInt = operand.name_hint
            break
    returnedType = "APInt"
    assert apInt

    return (
        IDNT
        + returnedType
        + " "
        + returnedValue
        + "("
        + apInt
        + ".getBitWidth(),"
        + lastReturn
        + ")"
        + END
    )


@lowerOperation.register
def _(op: SDivOp):
    return lowerToClassMethod(op, None, None)


@lowerOperation.register
def _(op: UDivOp):
    return lowerToClassMethod(op, None, None)


@lowerOperation.register
def _(op: SRemOp):
    return lowerToClassMethod(op, None, None)


@lowerOperation.register
def _(op: URemOp):
    return lowerToClassMethod(op, None, None)


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


def castToUnisgnedFromAPInt(operand: SSAValue | str) -> str:
    if isinstance(operand, str):
        return "(" + operand + ").getZExtValue()"
    elif isinstance(operand.type, TransIntegerType):
        return f"{operand.name_hint}.getZExtValue()"

    return str(operand.name_hint)


@lowerOperation.register
def _(op: SetHighBitsOp | SetLowBitsOp | ClearHighBitsOp | ClearLowBitsOp):
    return set_clear_bits(op)


@lowerOperation.register
def _(op: SetSignBitOp):
    returnedType = lowerType(op.results[0].type, op)
    returnedValue = get_ret_val(op)
    equals = EQ + get_operand(op, 0) + END + IDNT
    expr = returnedValue + get_op_str(op) + "("
    operands = ""
    expr = expr + operands + ")"

    return IDNT + returnedType + " " + returnedValue + equals + expr + END


@lowerOperation.register
def _(op: ClearSignBitOp):
    returnedType = lowerType(op.results[0].type, op)
    returnedValue = get_ret_val(op)
    equals = EQ + get_operand(op, 0) + END + IDNT
    expr = returnedValue + get_op_str(op) + "("
    operands = ""
    expr = expr + operands + ")"

    return IDNT + returnedType + " " + returnedValue + equals + expr + END


@lowerOperation.register
def _(op: GetLowBitsOp | GetHighBitsOp):
    return lowerToClassMethod(op, castToUnisgnedFromAPInt)


@lowerOperation.register
def _(op: GetBitWidthOp):
    return lowerToClassMethod(op, None, castToAPIntFromUnsigned)


@lowerOperation.register
def _(op: SMaxOp | SMinOp | UMaxOp | UMinOp):
    return lower_min_max(op)


def lower_min_max(op: UMinOp | UMaxOp | SMinOp | SMaxOp) -> str:
    returnedType = lowerType(op.operands[0].type, op)
    returnedValue = get_ret_val(op)
    operands = get_op_names(op)
    operator = get_op_str(op)

    expr = operator + "(" + operands[0] + "," + operands[1] + ")"

    return IDNT + returnedType + " " + returnedValue + EQ + expr + END


@lowerOperation.register
def _(op: ShlOp | AShrOp | LShrOp | ExtractOp):
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

    loopBefore = ""
    for i, blk_arg in enumerate(block_iter_args):
        iter_type = lowerType(iter_args[i].type, iter_args[i].owner)
        iter_name = blk_arg.name_hint
        iter_arg = iter_args[i].name_hint
        assert iter_name
        assert iter_arg

        loopBefore += IDNT + iter_type + " " + iter_name + EQ + iter_arg + END

    loopFor = IDNT + "for(APInt {0} = {1}; {0}.ule({2}); {0}+={3}){{\n".format(
        indvar.name_hint, lowerBound, upperBound, step
    )

    return loopBefore + loopFor


@lowerOperation.register
def _(op: NextLoopOp) -> str:
    loopBlock = op.parent_block()
    assert loopBlock
    _, *block_iter_args = loopBlock.args
    assignments = ""
    for i, arg in enumerate(op.operands):
        block_arg = block_iter_args[i].name_hint
        arg_name = arg.name_hint
        assert block_arg
        assert arg_name

        assignments += IDNT + block_arg + EQ + arg_name + END

    endLoopFor = IDNT + "}\n"
    loopOp = loopBlock.parent_op()
    assert loopOp

    for i, res in enumerate(loopOp.results):
        ty = lowerType(res.type, loopOp)
        res_name = res.name_hint
        block_arg = block_iter_args[i].name_hint
        assert res_name
        assert block_arg

        endLoopFor += IDNT + ty + " " + res_name + EQ + block_arg + END

    return assignments + endLoopFor


@lowerOperation.register
def _(op: RepeatOp):
    returnedType = lowerType(op.operands[0].type, op)
    returnedValue = get_ret_val(op)
    arg0_name = get_operand(op, 0)
    count = get_operand(op, 1)
    initExpr = IDNT + returnedType + " " + returnedValue + EQ + arg0_name + END
    forHead = (
        IDNT + "for(APInt i(" + count + ".getBitWidth(),1);i.ult(" + count + ");++i){\n"
    )
    forBody = (
        IDNT
        + IDNT
        + returnedValue
        + EQ
        + returnedValue
        + ".concat("
        + arg0_name
        + ")"
        + END
    )
    forEnd = IDNT + "}\n"
    return initExpr + forHead + forBody + forEnd


@lowerOperation.register
def _(op: AddPoisonOp):
    returnedType = lowerType(op.results[0].type)
    returnedValue = get_ret_val(op)
    operand = get_operand(op, 0)

    return IDNT + returnedType + " " + returnedValue + EQ + operand + END


@lowerOperation.register
def _(op: RemovePoisonOp) -> str:
    returnedType = lowerType(op.results[0].type)
    returnedValue = get_ret_val(op)
    operand = get_operand(op, 0)

    return IDNT + returnedType + " " + returnedValue + EQ + operand + END
