from xdsl.ir import Attribute, Dialect, Operation, OpResult
from xdsl.irdl import (OpAttr, Operand, irdl_op_definition)

from traits.effects import Pure


@irdl_op_definition
class Addf(Operation, Pure):
    name = "arith.addf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Addi(Operation, Pure):
    name = "arith.addi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class AdduiCarry(Operation, Pure):
    name = "arith.addui_carry"
    lhs: Operand
    rhs: Operand
    sum: OpResult
    carry: OpResult


@irdl_op_definition
class Andi(Operation, Pure):
    name = "arith.andi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Bitcast(Operation, Pure):
    name = "arith.bitcast"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Ceildivsi(Operation, Pure):
    name = "arith.ceildivsi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Ceildivui(Operation, Pure):
    name = "arith.ceildivui"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Cmpf(Operation, Pure):
    name = "arith.cmpf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Cmpi(Operation, Pure):
    name = "arith.cmpi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Constant(Operation, Pure):
    name = "arith.constant"
    value: OpAttr[Attribute]
    result: OpResult


@irdl_op_definition
class Divf(Operation, Pure):
    name = "arith.divf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Divsi(Operation, Pure):
    name = "arith.divsi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Divui(Operation, Pure):
    name = "arith.divui"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Extf(Operation, Pure):
    name = "arith.extf"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Extsi(Operation, Pure):
    name = "arith.extsi"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Extui(Operation, Pure):
    name = "arith.extui"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Fptosi(Operation, Pure):
    name = "arith.fptosi"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Fptoui(Operation, Pure):
    name = "arith.fptoui"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Floordivsi(Operation, Pure):
    name = "arith.floordivsi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class IndexCast(Operation, Pure):
    name = "arith.index_cast"
    _in: Operand
    out: OpResult


@irdl_op_definition
class IndexCastui(Operation, Pure):
    name = "arith.index_castui"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Maxf(Operation, Pure):
    name = "arith.maxf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Maxsi(Operation, Pure):
    name = "arith.maxsi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Maxui(Operation, Pure):
    name = "arith.maxui"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Minf(Operation, Pure):
    name = "arith.minf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Minsi(Operation, Pure):
    name = "arith.minsi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Minui(Operation, Pure):
    name = "arith.minui"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Mulf(Operation, Pure):
    name = "arith.mulf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Muli(Operation, Pure):
    name = "arith.muli"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Negf(Operation, Pure):
    name = "arith.negf"
    operand: Operand
    result: OpResult


@irdl_op_definition
class Ori(Operation, Pure):
    name = "arith.ori"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Remf(Operation, Pure):
    name = "arith.remf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Remsi(Operation, Pure):
    name = "arith.remsi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Remui(Operation, Pure):
    name = "arith.remui"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Sitofp(Operation, Pure):
    name = "arith.sitofp"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Shli(Operation, Pure):
    name = "arith.shli"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Shrsi(Operation, Pure):
    name = "arith.shrsi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Shrui(Operation, Pure):
    name = "arith.shrui"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Subf(Operation, Pure):
    name = "arith.subf"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Subi(Operation, Pure):
    name = "arith.subi"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Truncf(Operation, Pure):
    name = "arith.truncf"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Trunci(Operation, Pure):
    name = "arith.trunci"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Uitofp(Operation, Pure):
    name = "arith.uitofp"
    _in: Operand
    out: OpResult


@irdl_op_definition
class Xori(Operation, Pure):
    name = "arith.xori"
    lhs: Operand
    rhs: Operand
    result: OpResult


@irdl_op_definition
class Select(Operation, Pure):
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
