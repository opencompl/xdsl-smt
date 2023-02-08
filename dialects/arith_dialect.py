from xdsl.ir import Attribute, Dialect, Operation, OpResult
from xdsl.irdl import (OpAttr, Operand, irdl_op_definition)


@irdl_op_definition
class Addf(Operation):
    name = "arith.addf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Addi(Operation):
    name = "arith.addi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class AdduiCarry(Operation):
    name = "arith.addui_carry"
    lhs: Operand
    rhs: Operand
    sum: OpResult
    carry: OpResult


@irdl_op_definition
class Andi(Operation):
    name = "arith.andi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Bitcast(Operation):
    name = "arith.bitcast"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Ceildivsi(Operation):
    name = "arith.ceildivsi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Ceildivui(Operation):
    name = "arith.ceildivui"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Cmpf(Operation):
    name = "arith.cmpf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Cmpi(Operation):
    name = "arith.cmpi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Constant(Operation):
    name = "arith.constant"
    value: OpAttr[Attribute]
    result: OpResult


@irdl_op_definition
class Divf(Operation):
    name = "arith.divf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Divsi(Operation):
    name = "arith.divsi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Divui(Operation):
    name = "arith.divui"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Extf(Operation):
    name = "arith.extf"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Extsi(Operation):
    name = "arith.extsi"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Extui(Operation):
    name = "arith.extui"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Fptosi(Operation):
    name = "arith.fptosi"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Fptoui(Operation):
    name = "arith.fptoui"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Floordivsi(Operation):
    name = "arith.floordivsi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class IndexCast(Operation):
    name = "arith.index_cast"
    _in: Operand
    out: OpResult


@irdl_op_definition
class IndexCastui(Operation):
    name = "arith.index_castui"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Maxf(Operation):
    name = "arith.maxf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Maxsi(Operation):
    name = "arith.maxsi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Maxui(Operation):
    name = "arith.maxui"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Minf(Operation):
    name = "arith.minf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Minsi(Operation):
    name = "arith.minsi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Minui(Operation):
    name = "arith.minui"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Mulf(Operation):
    name = "arith.mulf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Muli(Operation):
    name = "arith.muli"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Negf(Operation):
    name = "arith.negf"
    operand: Operand
    result: OpResult


@irdl_op_definition
class Ori(Operation):
    name = "arith.ori"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Remf(Operation):
    name = "arith.remf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Remsi(Operation):
    name = "arith.remsi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Remui(Operation):
    name = "arith.remui"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Sitofp(Operation):
    name = "arith.sitofp"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Shli(Operation):
    name = "arith.shli"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Shrsi(Operation):
    name = "arith.shrsi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Shrui(Operation):
    name = "arith.shrui"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Subf(Operation):
    name = "arith.subf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Subi(Operation):
    name = "arith.subi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Truncf(Operation):
    name = "arith.truncf"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Trunci(Operation):
    name = "arith.trunci"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Uitofp(Operation):
    name = "arith.uitofp"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Xori(Operation):
    name = "arith.xori"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Select(Operation):
    name = "arith.select"
    condition: Operand
    true_value: Operand
    false_value: Operand
    result: OpResult

Arith = Dialect([
        Addf,
        Addi,
        AdduiCarry,
        Andi,
        Bitcast,
        Ceildivsi,
        Ceildivui,
        Cmpf,
        Cmpi,
        Constant,
        Divf,
        Divsi,
        Divui,
        Extf,
        Extsi,
        Extui,
        Fptosi,
        Fptoui,
        Floordivsi,
        IndexCast,
        IndexCastui,
        Maxf,
        Maxsi,
        Maxui,
        Minf,
        Minsi,
        Minui,
        Mulf,
        Muli,
        Negf,
        Ori,
        Remf,
        Remsi,
        Remui,
        Sitofp,
        Shli,
        Shrsi,
        Shrui,
        Subf,
        Subi,
        Truncf,
        Trunci,
        Uitofp,
        Xori,
        Select,
        ], [])