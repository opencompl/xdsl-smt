"""CLI tool: convert a transfer function (MLIR) to a PBV SMT-LIB soundness query."""

import argparse
import sys

from xdsl.context import Context
from xdsl.parser import Parser
from xdsl.dialects.func import Func, FuncOp, ReturnOp
from xdsl.dialects.builtin import Builtin, IntegerAttr, i1
from xdsl.dialects.arith import Arith, AndIOp, OrIOp, XOrIOp
from xdsl.dialects.arith import ConstantOp as ArithConstantOp
from xdsl.ir import SSAValue

from xdsl_smt.dialects.transfer import (
    Transfer,
    AbstractValueType,
    AndOp,
    OrOp,
    XorOp,
    AddOp,
    SubOp,
    MulOp,
    SDivOp,
    UDivOp,
    SRemOp,
    URemOp,
    ShlOp,
    AShrOp,
    LShrOp,
    NegOp,
    GetAllOnesOp,
    GetBitWidthOp,
    Constant as TransferConstant,
    SMinOp,
    SMaxOp,
    UMinOp,
    UMaxOp,
    CmpOp,
    SelectOp,
    GetOp,
    MakeOp,
    CountLZeroOp,
    CountLOneOp,
    CountRZeroOp,
    CountROneOp,
    PopCountOp,
)

# Maps CmpOp predicate index → SMT-LIB boolean expression builder
_CMP_PREDICATES: dict[int, object] = {
    0: lambda l, r: f"(= {l} {r})",
    1: lambda l, r: f"(not (= {l} {r}))",
    2: lambda l, r: f"(bvslt {l} {r})",
    3: lambda l, r: f"(bvsle {l} {r})",
    4: lambda l, r: f"(bvsgt {l} {r})",
    5: lambda l, r: f"(bvsge {l} {r})",
    6: lambda l, r: f"(bvult {l} {r})",
    7: lambda l, r: f"(bvule {l} {r})",
    8: lambda l, r: f"(bvugt {l} {r})",
    9: lambda l, r: f"(bvuge {l} {r})",
}


def _interpret_func(func_op: FuncOp, concrete_op: str) -> str:
    """Symbolically interpret *func_op* and return a PBV SMT-LIB soundness query."""

    # Maps SSA value → PBV expression string (bitvec or bool)
    val_to_expr: dict[SSAValue, str] = {}
    # Maps SSA value → list of PBV expression strings (one per component)
    val_to_components: dict[SSAValue, list[str]] = {}

    block = func_op.body.block

    # ── Process function arguments ────────────────────────────────────────────
    # abstract_args: (base_name, num_components) — one entry per abstract arg
    # concrete_vars: "IN0", "IN1", … — one per abstract arg
    abstract_args: list[tuple[str, int]] = []
    concrete_vars: list[str] = []

    for arg in block.args:
        if isinstance(arg.type, AbstractValueType):
            idx = len(concrete_vars)
            concrete_vars.append(f"IN{idx}")
            name_hint = arg.name_hint
            base = name_hint.upper() if name_hint else f"ARG{idx}"
            n = arg.type.get_num_fields()
            val_to_components[arg] = [f"{base}{i}" for i in range(n)]
            abstract_args.append((base, n))
        else:
            # Plain !transfer.integer or i1 argument
            name_hint = arg.name_hint
            var = name_hint.upper() if name_hint else f"SCALAR{len(val_to_expr)}"
            val_to_expr[arg] = var

    # ── Walk operations ───────────────────────────────────────────────────────
    result_components: list[str] | None = None

    for op in block.ops:
        # ── Terminator ──────────────────────────────────────────────────────
        if isinstance(op, ReturnOp):
            ret = op.operands[0]
            if ret in val_to_components:
                result_components = val_to_components[ret]
            else:
                result_components = [val_to_expr[ret]]

        # ── Abstract-value constructors / destructors ────────────────────────
        elif isinstance(op, GetOp):
            operand = op.operands[0]
            idx = op.index.value.data
            val_to_expr[op.result] = val_to_components[operand][idx]

        elif isinstance(op, MakeOp):
            val_to_components[op.result] = [
                val_to_expr[operand] for operand in op.operands
            ]

        # ── Binary bitvec ops ────────────────────────────────────────────────
        elif isinstance(op, AndOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(bvand {l} {r})"

        elif isinstance(op, OrOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(bvor {l} {r})"

        elif isinstance(op, XorOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(bvxor {l} {r})"

        elif isinstance(op, AddOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(bvadd {l} {r})"

        elif isinstance(op, SubOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(bvsub {l} {r})"

        elif isinstance(op, MulOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(bvmul {l} {r})"

        elif isinstance(op, ShlOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(bvshl {l} {r})"

        elif isinstance(op, LShrOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(bvlshr {l} {r})"

        elif isinstance(op, AShrOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(bvashr {l} {r})"

        elif isinstance(op, SDivOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(bvsdiv {l} {r})"

        elif isinstance(op, UDivOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(bvudiv {l} {r})"

        elif isinstance(op, SRemOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(bvsrem {l} {r})"

        elif isinstance(op, URemOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(bvurem {l} {r})"

        # ── Unary bitvec ops ─────────────────────────────────────────────────
        elif isinstance(op, NegOp):
            o = val_to_expr[op.operands[0]]
            val_to_expr[op.result] = f"(bvnot {o})"

        elif isinstance(op, GetAllOnesOp):
            val_to_expr[op.result] = "(bvnot (int_to_pbv k 0))"

        elif isinstance(op, GetBitWidthOp):
            val_to_expr[op.result] = "(int_to_pbv k k)"

        elif isinstance(op, TransferConstant):
            value = op.value.value.data
            val_to_expr[op.result] = f"(int_to_pbv k {value})"

        # ── Min / max ────────────────────────────────────────────────────────
        elif isinstance(op, SMinOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(ite (bvsle {l} {r}) {l} {r})"

        elif isinstance(op, SMaxOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(ite (bvsge {l} {r}) {l} {r})"

        elif isinstance(op, UMinOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(ite (bvule {l} {r}) {l} {r})"

        elif isinstance(op, UMaxOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            val_to_expr[op.result] = f"(ite (bvuge {l} {r}) {l} {r})"

        # ── Comparison (returns i1 / Bool) ───────────────────────────────────
        elif isinstance(op, CmpOp):
            l, r = val_to_expr[op.operands[0]], val_to_expr[op.operands[1]]
            pred = op.attributes["predicate"].value.data
            builder = _CMP_PREDICATES.get(pred)
            if builder is None:
                raise NotImplementedError(f"Unsupported CmpOp predicate index: {pred}")
            val_to_expr[op.result] = builder(l, r)  # type: ignore[operator]

        # ── Select (i1 condition) ────────────────────────────────────────────
        elif isinstance(op, SelectOp):
            cond = val_to_expr[op.operands[0]]
            tv = val_to_expr[op.operands[1]]
            fv = val_to_expr[op.operands[2]]
            val_to_expr[op.result] = f"(ite {cond} {tv} {fv})"

        # ── arith ops on i1 (boolean logic) ─────────────────────────────────
        elif isinstance(op, ArithConstantOp):
            value = op.value.value.data
            if op.result.type == i1:
                val_to_expr[op.result] = "true" if value != 0 else "false"
            else:
                val_to_expr[op.result] = f"(int_to_pbv k {value})"

        elif isinstance(op, AndIOp):
            l, r = val_to_expr[op.lhs], val_to_expr[op.rhs]
            val_to_expr[op.result] = f"(and {l} {r})"

        elif isinstance(op, OrIOp):
            l, r = val_to_expr[op.lhs], val_to_expr[op.rhs]
            val_to_expr[op.result] = f"(or {l} {r})"

        elif isinstance(op, XOrIOp):
            l, r = val_to_expr[op.lhs], val_to_expr[op.rhs]
            val_to_expr[op.result] = f"(xor {l} {r})"

        # ── Unsupported ops ──────────────────────────────────────────────────
        elif isinstance(
            op, (CountLZeroOp, CountLOneOp, CountRZeroOp, CountROneOp, PopCountOp)
        ):
            raise NotImplementedError(
                f"Operation {op.name!r} has no parametric bitvector equivalent "
                f"and is not supported in PBV mode."
            )

        # (Other ops — e.g. arith.subi — are ignored if not encountered;
        #  unknown ops that produce used results will surface naturally as a
        #  KeyError in val_to_expr later.)

    if result_components is None:
        raise ValueError("Transfer function has no 'func.return' terminator.")

    # ── Emit SMT-LIB ─────────────────────────────────────────────────────────
    zero = "(int_to_pbv k 0)"
    concrete_app = f"({concrete_op} {' '.join(concrete_vars)})"

    lines: list[str] = []
    lines.append("(set-logic ALL)")
    lines.append("(declare-const k Int)")
    lines.append("")

    # Concrete input variables
    for var in concrete_vars:
        lines.append(f"(declare-fun {var} () (_ BitVec k))")
    lines.append("")

    # Abstract input component variables
    for base, n in abstract_args:
        for i in range(n):
            lines.append(f"(declare-fun {base}{i} () (_ BitVec k))")
    lines.append("")

    # Build premise: γ(k0,k1) ∋ INi  +  well-formedness k0&k1 = 0
    premise_parts: list[str] = []
    for (base, _n), cv in zip(abstract_args, concrete_vars):
        k0, k1 = f"{base}0", f"{base}1"
        premise_parts.append(f"(= (bvand {cv} {k0}) {zero})")
        premise_parts.append(f"(= (bvand {cv} {k1}) {k1})")
    for base, _n in abstract_args:
        k0, k1 = f"{base}0", f"{base}1"
        premise_parts.append(f"(= (bvand {k0} {k1}) {zero})")

    # Build conclusion: γ(res0,res1) ∋ concrete_app
    res0 = result_components[0]
    res1 = result_components[1] if len(result_components) > 1 else result_components[0]
    conclusion_parts: list[str] = [
        f"(= (bvand {concrete_app} {res0}) {zero})",
        f"(= (bvand {concrete_app} {res1}) {res1})",
    ]

    def _join(parts: list[str], indent: int) -> str:
        pad = " " * indent
        if len(parts) == 1:
            return parts[0]
        inner = f"\n{pad}".join(parts)
        return f"(and\n{pad}{inner}\n{' ' * (indent - 2)})"

    lines.append("(assert (not")
    lines.append("  (=>")
    lines.append(f"    {_join(premise_parts, 6)}")
    lines.append(f"    {_join(conclusion_parts, 6)}")
    lines.append("  )")
    lines.append("))")
    lines.append("(check-sat)")

    return "\n".join(lines)


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="Generate a PBV SMT-LIB soundness query from a transfer function."
    )
    arg_parser.add_argument(
        "input_file",
        nargs="?",
        help="Path to the MLIR file (reads from stdin if omitted).",
    )
    arg_parser.add_argument(
        "--concrete-op",
        "-c",
        required=True,
        help="SMT-LIB concrete operation name (e.g. bvxor, bvadd).",
    )
    args = arg_parser.parse_args()

    if args.input_file is None:
        source = sys.stdin.read()
        filename = "<stdin>"
    else:
        with open(args.input_file) as f:
            source = f.read()
        filename = args.input_file

    ctx = Context()
    ctx.load_dialect(Transfer)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(Arith)

    parser = Parser(ctx, source, filename)
    func_op = parser.parse_op()
    if not isinstance(func_op, FuncOp):
        print(
            f"error: expected a func.func operation, got {type(func_op).__name__}",
            file=sys.stderr,
        )
        sys.exit(1)

    result = _interpret_func(func_op, args.concrete_op)
    print(result)


if __name__ == "__main__":
    main()
