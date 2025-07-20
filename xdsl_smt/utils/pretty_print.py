from typing import Mapping, IO

from xdsl.ir import SSAValue, OpResult
from xdsl.utils.hints import isa
from xdsl_smt.traits.smt_printer import SimpleSMTLibOp

from xdsl.dialects.builtin import IntegerAttr, IntegerType
from xdsl_smt.dialects import (
    smt_dialect as smt,
    smt_bitvector_dialect as bv,
    synth_dialect as synth,
)


def pretty_print_value(
    x: SSAValue,
    nested: bool,
    names: Mapping[SSAValue, str] = {},
    *,
    file: IO[str],
):
    infix = isinstance(x, OpResult) and len(x.op.operand_types) > 1
    if x in names:
        print(names[x], end="", file=file)
        return

    parenthesized = infix and nested
    if parenthesized:
        print("(", end="", file=file)
    match x:
        case OpResult(op=synth.ConstantOp(), type=smt.BoolType()):
            print("cst", end="", file=file)
        case OpResult(op=synth.ConstantOp(), type=bv.BitVectorType(width=width)):
            print(f"cst#{width.data}", end="", file=file)
        case OpResult(op=smt.ConstantBoolOp(value=val), index=0):
            print("⊤" if val else "⊥", end="", file=file)
        case OpResult(op=bv.ConstantOp(value=val), index=0):
            width = val.type.width.data
            value = val.value.data
            if width % 4 == 0:
                print(f"0x{value:0{width // 4}x}", end="", file=file)
            else:
                print(f"{{:0{width}b}}".format(value), end="", file=file)
        case OpResult(op=smt.NotOp(arg=arg), index=0):
            print("¬", end="", file=file)
            pretty_print_value(arg, True, names, file=file)
        case OpResult(op=smt.AndOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, names, file=file)
            print(" ∧ ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(op=smt.OrOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, names, file=file)
            print(" ∨ ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(op=smt.ImpliesOp(lhs=lhs, rhs=rhs), index=0):
            pretty_print_value(lhs, True, names, file=file)
            print(" → ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(op=smt.DistinctOp(lhs=lhs, rhs=rhs), index=0):
            pretty_print_value(lhs, True, names, file=file)
            print(" ≠ ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(op=smt.EqOp(lhs=lhs, rhs=rhs), index=0):
            pretty_print_value(lhs, True, names, file=file)
            print(" = ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(op=smt.XOrOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, names, file=file)
            print(" ⊕ ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(
            op=smt.IteOp(cond=cond, true_val=true_val, false_val=false_val), index=0
        ):
            pretty_print_value(cond, True, names, file=file)
            print(" ? ", end="", file=file)
            pretty_print_value(true_val, True, names, file=file)
            print(" : ", end="", file=file)
            pretty_print_value(false_val, True, names, file=file)
        case OpResult(op=bv.AddOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, names, file=file)
            print(" + ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(op=bv.AndOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, names, file=file)
            print(" & ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(op=bv.OrOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, names, file=file)
            print(" | ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(op=bv.XorOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, names, file=file)
            print(" ^ ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(op=bv.MulOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, names, file=file)
            print(" * ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(op=bv.NotOp(arg=arg), index=0):
            print("~", end="", file=file)
            pretty_print_value(arg, True, names, file=file)
        case OpResult(op=bv.ShlOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, names, file=file)
            print(" << ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(op=bv.AShrOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, names, file=file)
            print(" a>> ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(op=bv.LShrOp(operands=(lhs, rhs)), index=0):
            pretty_print_value(lhs, True, names, file=file)
            print(" l>> ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(op=bv.NegOp(operands=(arg,)), index=0):
            print("-", end="", file=file)
            pretty_print_value(arg, True, names, file=file)
        case OpResult(
            op=bv.CmpOp(operands=(lhs, rhs), properties={"pred": pred}), index=0
        ):
            assert isa(pred, IntegerAttr[IntegerType])
            pretty_print_value(lhs, True, names, file=file)
            predicates = {
                0: "s<",
                1: "s<=",
                2: "s>",
                3: "s>=",
                4: "u<",
                5: "u<=",
                6: "u>",
                7: "u>=",
            }
            print(f" {predicates[pred.value.data]} ", end="", file=file)
            pretty_print_value(rhs, True, names, file=file)
        case OpResult(op=op, index=0) if (
            isinstance(op, SimpleSMTLibOp) and len(op.operands) == 2
        ):
            pretty_print_value(op.operands[0], True, names, file=file)
            print(f" {op.op_name()} ", end="", file=file)
            pretty_print_value(op.operands[1], True, names, file=file)
        case _:
            raise ValueError(f"Unknown value for pretty print: {x}")
    if parenthesized:
        print(")", end="", file=file)
