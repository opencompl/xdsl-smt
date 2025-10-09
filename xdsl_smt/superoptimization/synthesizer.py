"""
This files defines a function that synthesize constants in an RHS
program such that it is a refinement of an LHS program.
"""

import z3  # type: ignore[reportMissingTypeStubs]
from typing import Sequence, Mapping, cast, Any
from xdsl.context import Context
from xdsl.ir import SSAValue, Attribute, OperationInvT
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.builder import Builder
from xdsl.utils.hints import isa

from xdsl.dialects import arith
from xdsl.dialects.builtin import (
    ModuleOp,
    StringAttr,
    IntegerAttr,
    IntegerType,
)
from xdsl_smt.dialects.smt_dialect import (
    DeclareConstOp,
    DefineFunOp,
    AssertOp,
    CheckSatOp,
    BoolType,
)
from xdsl_smt.dialects import (
    synth_dialect as synth,
    smt_utils_dialect as smt_utils,
    smt_bitvector_dialect as smt_bv,
)
from xdsl.dialects.func import FuncOp
from xdsl.dialects.builtin import ArrayAttr
from xdsl_smt.dialects.smt_bitvector_dialect import BitVectorAttr
from xdsl_smt.semantics.semantics import OperationSemantics
from xdsl_smt.semantics.refinements import (
    insert_function_refinement_with_forall,
)

from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl_smt.passes.lower_to_smt import LowerToSMTPass
from xdsl_smt.passes.lower_to_smt.smt_lowerer import SMTLowerer
from xdsl_smt.passes.lower_memory_effects import LowerMemoryEffectsPass
from xdsl.transforms.common_subexpression_elimination import (
    CommonSubexpressionElimination,
)
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl_smt.passes.lower_effects_with_memory import LowerEffectsWithMemoryPass
from xdsl_smt.passes.lower_memory_to_array import LowerMemoryToArrayPass
from xdsl_smt.passes.smt_expand import SMTExpand
from xdsl_smt.passes.transfer_inline import FunctionCallInline
from xdsl_smt.utils.run_with_smt_solver import run_module_through_smtlib


class SynthSemantics(OperationSemantics):
    def get_semantics(
        self,
        operands: Sequence[SSAValue],
        results: Sequence[Attribute],
        attributes: Mapping[str, Attribute | SSAValue],
        effect_state: SSAValue | None,
        rewriter: PatternRewriter,
    ) -> tuple[Sequence[SSAValue], SSAValue | None]:
        result_type = SMTLowerer.lower_type(results[0])
        declare_const = DeclareConstOp(result_type)
        res = rewriter.insert(declare_const).res
        conv_res = rewriter.insert(synth.ConversionOp(res, results[0])).results[0]
        assert "cst_name" in attributes
        assert isinstance(attributes["cst_name"], Attribute)
        from_value_op = rewriter.insert(synth.FromValueOp(conv_res))
        from_value_op.attributes["cst_name"] = attributes["cst_name"]
        return ((res,), effect_state)


def z3_value_to_attribute(val: z3.ExprRef) -> Attribute:
    if isinstance(val, z3.DatatypeRef):
        if val.decl().name() == "pair":
            first = z3_value_to_attribute(
                val.arg(0)  # pyright: ignore[reportUnknownMemberType]
            )
            second = z3_value_to_attribute(
                val.arg(1)  # pyright: ignore[reportUnknownMemberType]
            )
            return ArrayAttr([first, second])
    if isinstance(val, z3.BitVecNumRef):
        return BitVectorAttr(int(val.as_long()), val.size())
    if isinstance(val, z3.BoolRef):
        return BitVectorAttr(1 if z3.is_true(val) else 0, 1)
    raise ValueError(
        f"No known conversion of Z3 value of type {type(val)} to Attribute."
    )


def move_synth_constants_at_toplevel(
    module: ModuleOp, insert_point: InsertPoint
) -> None:
    """Move synth.constant operations to the beginning of the module."""
    builder = Builder(insert_point)

    for op in module.walk():
        if isinstance(op, synth.ConstantOp):
            op.detach()
            builder.insert(op)


def optimize_module(ctx: Context, module: ModuleOp, with_pairs: bool = True) -> None:
    CanonicalizePass().apply(ctx, module)
    if with_pairs:
        LowerPairs().apply(ctx, module)
    CanonicalizePass().apply(ctx, module)
    CommonSubexpressionElimination().apply(ctx, module)
    CanonicalizePass().apply(ctx, module)


def get_op_from_module(module: ModuleOp, opT: type[OperationInvT]) -> OperationInvT:
    """
    Get an operation of a specific type from a module.
    """
    for op in module.ops:
        if isinstance(op, opT):
            return op
    raise ValueError(f"No operation of type '{opT.name}' found in the module.")


def assign_names_to_synth_constants(module: ModuleOp) -> dict[str, synth.ConstantOp]:
    name_to_value = dict[str, synth.ConstantOp]()
    counter = 0
    for op in module.walk():
        if isinstance(op, synth.ConstantOp):
            op.attributes["cst_name"] = StringAttr(f"__synth_cst_{counter}__")
            name_to_value[f"__synth_cst_{counter}__"] = op
            counter += 1
    return name_to_value


def assign_name_hints_to_declare_const(module: ModuleOp) -> dict[str, SSAValue]:
    """
    Assign name hints to all DeclareConstOp in the module.
    """
    mapping: dict[str, SSAValue] = {}
    counter = 0
    for op in module.walk():
        if isinstance(op, DeclareConstOp):
            op.res.name_hint = f"__synth_cst_{counter}__"
            mapping[op.res.name_hint] = op.res
            counter += 1
    return mapping


class PoisonValue:
    """A marker for a poison value."""

    pass


def compute_value(
    module: ModuleOp, value: SSAValue, ssavalue_to_value: dict[SSAValue, Any]
) -> Any:
    if value in ssavalue_to_value:
        return ssavalue_to_value[value]
    if isinstance(value.owner, synth.ConversionOp):
        inner_value = compute_value(module, value.owner.input, ssavalue_to_value)
        if isa(
            value.owner.input.type, smt_utils.PairType[smt_bv.BitVectorType, BoolType]
        ) and isinstance(value.type, IntegerType):
            assert isa(inner_value, ArrayAttr)
            int_value, poison_marker = inner_value.data
            assert isinstance(int_value, BitVectorAttr)
            assert isinstance(poison_marker, BitVectorAttr)
            if poison_marker.value.data != 0:
                return PoisonValue()
            return IntegerAttr.from_int_and_width(
                int_value.value.data, int_value.type.width.data
            )
        raise ValueError(
            f"Unsupported conversion from type {value.owner.input.type} to {value.type}"
        )
    if isinstance(value.owner, smt_utils.PairOp):
        first = compute_value(module, value.owner.first, ssavalue_to_value)
        second = compute_value(module, value.owner.second, ssavalue_to_value)
        return ArrayAttr([first, second])
    raise ValueError(f"Unsupported computation of operation {value.owner}")


def replace_synth_constant_with_op(synth_const: synth.ConstantOp, value: Any) -> None:
    rewriter = Rewriter()
    if isa(value, IntegerAttr):
        new_op = arith.ConstantOp(value)
        rewriter.replace_op(synth_const, [new_op])


def synthesize_constants(
    lhs: ModuleOp,
    rhs: ModuleOp,
    ctx: Context,
    optimize: bool,
    timeout: int | None = None,
) -> ModuleOp | None:
    rhs_old = rhs
    # Give a name to each synth.constant so we can track them during the pipeline.
    name_to_synth_const = assign_names_to_synth_constants(rhs_old)

    rhs = rhs_old.clone()

    func_lhs = get_op_from_module(lhs, FuncOp)
    assert isinstance(func_lhs, FuncOp)
    lhs_func_type = func_lhs.function_type

    # Move smt.synth.constant to function arguments
    func_rhs = get_op_from_module(rhs, FuncOp)
    assert isinstance(func_rhs, FuncOp)
    rhs_func_type = func_rhs.function_type

    # Move synth.constant outside of the rhs function body
    move_synth_constants_at_toplevel(rhs, InsertPoint.at_start(rhs.body.block))

    SMTLowerer.op_semantics[synth.ConstantOp] = SynthSemantics()
    # Convert both module to SMTLib
    LowerToSMTPass().apply(ctx, lhs)
    LowerToSMTPass().apply(ctx, rhs)

    func = get_op_from_module(lhs, DefineFunOp)
    func_rhs = get_op_from_module(rhs, DefineFunOp)

    # Combine both modules into a new one
    new_module = ModuleOp([])
    block = new_module.body.blocks[0]
    for op in lhs.body.ops:
        op.detach()
        block.add_op(op)
    for op in rhs.body.ops:
        op.detach()
        block.add_op(op)

    if optimize:
        optimize_module(ctx, new_module, with_pairs=False)

    LowerMemoryEffectsPass().apply(ctx, new_module)

    if optimize:
        optimize_module(ctx, new_module, with_pairs=False)

    LowerEffectsWithMemoryPass().apply(ctx, new_module)

    if optimize:
        optimize_module(ctx, new_module, with_pairs=False)

    new_module.verify()
    refinement = insert_function_refinement_with_forall(
        func,
        lhs_func_type,
        func_rhs,
        rhs_func_type,
        InsertPoint.at_end(block),
    )
    block.add_op(AssertOp(refinement))
    new_module.verify()

    if optimize:
        optimize_module(ctx, new_module)

    move_synth_constants_at_toplevel(
        new_module, InsertPoint.at_start(new_module.body.blocks[0])
    )

    if optimize:
        optimize_module(ctx, new_module)

    FunctionCallInline(True, {}).apply(ctx, new_module)
    for op in new_module.body.ops:
        if isinstance(op, DefineFunOp):
            new_module.body.block.erase_op(op)

    if optimize:
        optimize_module(ctx, new_module)

    # Lower memory to arrays
    LowerMemoryToArrayPass().apply(ctx, new_module)

    if optimize:
        optimize_module(ctx, new_module)

    # Expand ops not supported by all SMT solvers
    SMTExpand().apply(ctx, new_module)
    if optimize:
        optimize_module(ctx, new_module)

    name_to_ssavalue = assign_name_hints_to_declare_const(new_module)

    block.add_op(CheckSatOp())
    if timeout is not None:
        result, solver = run_module_through_smtlib(new_module, timeout=timeout)
    else:
        result, solver = run_module_through_smtlib(new_module)
    if result != z3.sat:
        return None

    model = solver.model()
    ssavalue_to_attr: dict[SSAValue, Attribute] = {}
    for d in cast(list[Any], model.decls()):
        # We remove the $ that is added when printing SMTLib
        name = d.name()[1:]
        attr = z3_value_to_attribute(model[d])  # type: ignore[reportUnknownMemberType]
        ssavalue_to_attr[name_to_ssavalue[name]] = attr

    name_to_from_value: dict[str, synth.FromValueOp] = {}
    for op in new_module.ops:
        if isinstance(op, synth.FromValueOp):
            assert "cst_name" in op.attributes
            assert isinstance(op.attributes["cst_name"], StringAttr)
            name_to_from_value[op.attributes["cst_name"].data] = op

    synth_const_to_values: dict[synth.ConstantOp, Any] = {}
    for name, synth_const in name_to_synth_const.items():
        input_value = name_to_from_value[name].input
        value = compute_value(new_module, input_value, ssavalue_to_attr)
        synth_const_to_values[synth_const] = value

    for synth_const, value in synth_const_to_values.items():
        replace_synth_constant_with_op(synth_const, value)

    return rhs_old
