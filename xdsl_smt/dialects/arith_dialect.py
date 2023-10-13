from xdsl.ir import Attribute, Dialect, OpResult
from xdsl.irdl import (
    operand_def,
    result_def,
    attr_def,
    Operand,
    irdl_op_definition,
    IRDLOperation,
)
from xdsl.parser import AnyIntegerAttr

from ..traits.effects import Pure


@irdl_op_definition
class Addf(IRDLOperation, Pure):
    name = "arith.addf"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Addi(IRDLOperation, Pure):
    name = "arith.addi"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class AdduiCarry(IRDLOperation, Pure):
    name = "arith.addui_carry"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    sum: OpResult = result_def()
    carry: OpResult = result_def()


@irdl_op_definition
class Andi(IRDLOperation, Pure):
    name = "arith.andi"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Bitcast(IRDLOperation, Pure):
    name = "arith.bitcast"
    in_: Operand = operand_def()
    out: OpResult = result_def()


@irdl_op_definition
class Ceildivsi(IRDLOperation, Pure):
    name = "arith.ceildivsi"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Ceildivui(IRDLOperation, Pure):
    name = "arith.ceildivui"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Cmpf(IRDLOperation, Pure):
    name = "arith.cmpf"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Cmpi(IRDLOperation, Pure):
    name = "arith.cmpi"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()

    predicate: AnyIntegerAttr = attr_def(AnyIntegerAttr)


@irdl_op_definition
class Constant(IRDLOperation, Pure):
    name = "arith.constant"
    value: Attribute = attr_def(Attribute)
    result: OpResult = result_def()


@irdl_op_definition
class Divf(IRDLOperation, Pure):
    name = "arith.divf"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Divsi(IRDLOperation, Pure):
    name = "arith.divsi"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Divui(IRDLOperation, Pure):
    name = "arith.divui"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Extf(IRDLOperation, Pure):
    name = "arith.extf"
    in_: Operand = operand_def()
    out: OpResult = result_def()


@irdl_op_definition
class Extsi(IRDLOperation, Pure):
    name = "arith.extsi"
    in_: Operand = operand_def()
    out: OpResult = result_def()


@irdl_op_definition
class Extui(IRDLOperation, Pure):
    name = "arith.extui"
    in_: Operand = operand_def()
    out: OpResult = result_def()


@irdl_op_definition
class Fptosi(IRDLOperation, Pure):
    name = "arith.fptosi"
    in_: Operand = operand_def()
    out: OpResult = result_def()


@irdl_op_definition
class Fptoui(IRDLOperation, Pure):
    name = "arith.fptoui"
    in_: Operand = operand_def()
    out: OpResult = result_def()


@irdl_op_definition
class Floordivsi(IRDLOperation, Pure):
    name = "arith.floordivsi"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class IndexCast(IRDLOperation, Pure):
    name = "arith.index_cast"
    in_: Operand = operand_def()
    out: OpResult = result_def()


@irdl_op_definition
class IndexCastui(IRDLOperation, Pure):
    name = "arith.index_castui"
    in_: Operand = operand_def()
    out: OpResult = result_def()


@irdl_op_definition
class Maxf(IRDLOperation, Pure):
    name = "arith.maxf"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Maxsi(IRDLOperation, Pure):
    name = "arith.maxsi"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Maxui(IRDLOperation, Pure):
    name = "arith.maxui"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Minf(IRDLOperation, Pure):
    name = "arith.minf"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Minsi(IRDLOperation, Pure):
    name = "arith.minsi"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Minui(IRDLOperation, Pure):
    name = "arith.minui"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Mulf(IRDLOperation, Pure):
    name = "arith.mulf"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Muli(IRDLOperation, Pure):
    name = "arith.muli"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Negf(IRDLOperation, Pure):
    name = "arith.negf"
    operand: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Ori(IRDLOperation, Pure):
    name = "arith.ori"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Remf(IRDLOperation, Pure):
    name = "arith.remf"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Remsi(IRDLOperation, Pure):
    name = "arith.remsi"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Remui(IRDLOperation, Pure):
    name = "arith.remui"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Sitofp(IRDLOperation, Pure):
    name = "arith.sitofp"
    in_: Operand = operand_def()
    out: OpResult = result_def()


@irdl_op_definition
class Shli(IRDLOperation, Pure):
    name = "arith.shli"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Shrsi(IRDLOperation, Pure):
    name = "arith.shrsi"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Shrui(IRDLOperation, Pure):
    name = "arith.shrui"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Subf(IRDLOperation, Pure):
    name = "arith.subf"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Subi(IRDLOperation, Pure):
    name = "arith.subi"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Truncf(IRDLOperation, Pure):
    name = "arith.truncf"
    in_: Operand = operand_def()
    out: OpResult = result_def()


@irdl_op_definition
class Trunci(IRDLOperation, Pure):
    name = "arith.trunci"
    in_: Operand = operand_def()
    out: OpResult = result_def()


@irdl_op_definition
class Uitofp(IRDLOperation, Pure):
    name = "arith.uitofp"
    in_: Operand = operand_def()
    out: OpResult = result_def()


@irdl_op_definition
class Xori(IRDLOperation, Pure):
    name = "arith.xori"
    lhs: Operand = operand_def()
    rhs: Operand = operand_def()
    result: OpResult = result_def()


@irdl_op_definition
class Select(IRDLOperation, Pure):
    name = "arith.select"
    condition: Operand = operand_def()
    true_value: Operand = operand_def()
    false_value: Operand = operand_def()
    result: OpResult = result_def()


Arith = Dialect(
    [
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
    ],
    [],
)
