from typing import cast
from xdsl.ir import Attribute, MLContext, OpResult, Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern, op_type_rewrite_pattern
from xdsl.dialects.builtin import IntegerAttr, IntegerType, ModuleOp, FunctionType
from xdsl.dialects.func import FuncOp, Return
from xdsl.passes import ModulePass

import dialects.smt_bitvector_dialect as bv_dialect
import dialects.arith_dialect as arith
from dialects.smt_bitvector_dialect import BitVectorType
from traits.smt_printer import SMTLibSort
from dialects.smt_utils_dialect import AnyPairType, FirstOp, PairOp, PairType, SecondOp
from dialects.smt_dialect import BoolType, ConstantBoolOp, DefineFunOp, ReturnOp, OrOp


def get_constant_bv_ops(value: int, width: int) -> list[Operation]:
    """
    Create operations returning a bitvector constant,
    as well as the poison indicator.
    """
    constant = bv_dialect.ConstantOp.from_int_value(value, width)
    poison = ConstantBoolOp.from_bool(False)
    pair = PairOp.from_values(constant.res, poison.res)
    return [constant, poison, pair]


def get_value_and_poison(
        value: SSAValue) -> tuple[list[Operation], SSAValue, SSAValue]:
    """
    Create a list of operations that extract the value and poison indicator
    of a converted integer type. Also return the created operations to
    extract these values.
    """
    value_op = FirstOp.from_value(value)
    poison_op = SecondOp.from_value(value)
    return [value_op, poison_op], value_op.res, poison_op.res


def convert_type(type: Attribute) -> Attribute:
    """Convert a type to an SMT sort"""
    if isinstance(type, IntegerType):
        return PairType(BitVectorType.from_int(type.width.data), BoolType())
    if isinstance(type, SMTLibSort):
        return type
    raise Exception("Cannot convert {type} attribute")


def convert_constant(
        op: arith.Constant
) -> tuple[list[Operation], list[OpResult], SSAValue]:
    constant_value = cast(IntegerAttr[IntegerType], op.value)
    ops = get_constant_bv_ops(constant_value.value.data,
                              constant_value.typ.width.data)
    new_res_val = [ops[-1].results[0]]
    ops.append(true_op := ConstantBoolOp.from_bool(False))
    return ops, new_res_val, true_op.res


def convert_ori(
    op: arith.Ori, ssa_mapping: dict[SSAValue, SSAValue]
) -> tuple[list[Operation], list[OpResult], SSAValue]:
    lhs_deconstruct, lhs_val, lhs_poison = get_value_and_poison(
        ssa_mapping[op.lhs])
    rhs_deconstruct, rhs_val, rhs_poison = get_value_and_poison(
        ssa_mapping[op.rhs])

    res_val_op = bv_dialect.OrOp.get(lhs_val, rhs_val)
    res_poison_op = OrOp.get(lhs_poison, rhs_poison)
    res_op = PairOp.from_values(res_val_op.res, res_poison_op.res)

    true_op = ConstantBoolOp.from_bool(False)
    new_ops = (lhs_deconstruct + rhs_deconstruct +
               [res_val_op, res_poison_op, res_op, true_op])

    return new_ops, [res_op.res], true_op.res


def convert_op(
        op: Operation,
        ssa_mapping: dict[SSAValue,
                          SSAValue]) -> tuple[list[Operation], SSAValue]:
    """
    Convert an arith operation to SMT.
    The new operations are returned, as well as an SSA value indicating
    if the operation triggered undefined behavior.
    """
    if isinstance(op, arith.Constant):
        new_ops, new_res, ub_value = convert_constant(op)
    elif isinstance(op, arith.Ori):
        new_ops, new_res, ub_value = convert_ori(op, ssa_mapping)
    else:
        raise Exception(f"Cannot convert '{op.name}' operation")

    for res, new in zip(op.results, new_res):
        ssa_mapping[res] = new

    return new_ops, ub_value


def merge_values_with_pairs(
        vals: list[SSAValue]) -> tuple[list[Operation], SSAValue]:
    """Merge a nonempty list of SSAValues into a single SSAValue (using smt.utils.pair)"""
    assert len(vals) > 0

    res: SSAValue = vals[-1]
    new_ops = list[Operation]()
    for i in reversed(range(len(vals) - 1)):
        new_op = PairOp.from_values(vals[i], res)
        new_ops.append(new_op)
        res = new_op.res
    return new_ops, res


def merge_types_with_pairs(types: list[Attribute]) -> Attribute:
    """Merge a nonempty list of types into a single type (using smt.utils.pair)"""
    assert len(types) > 0

    res: Attribute = types[-1]
    for i in reversed(range(len(types) - 1)):
        res = AnyPairType(types[i], res)
    return res


class FuncToSMTPattern(RewritePattern):
    """Convert func.func to an SMT formula"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter):
        """
        Convert a `func` function to an smt function.
        All `iN` arguments are translated to `pair<bv<N>, bool>`, the first
        value being the actual value of the integer, and the second one being
        a poison indicator.
        Functions also return an additional argument, representing if undefined
        behavior was triggered.
        """
        # We only handle single-block regions for now
        assert len(op.body.blocks) == 1

        # Mapping from old SSAValue to the new ones
        ssa_mapping = dict[SSAValue, SSAValue]()

        operand_types = [
            convert_type(input) for input in op.function_type.inputs.data
        ]
        result_types = merge_types_with_pairs(
            [BoolType()] +
            [convert_type(output) for output in op.function_type.outputs.data])

        # The SMT function replacing the func.func function
        smt_func = DefineFunOp.from_function_type(
            FunctionType.from_lists(operand_types, [result_types]),
            op.sym_name)

        for i, arg in enumerate(smt_func.body.blocks[0].args):
            ssa_mapping[op.body.blocks[0].args[i]] = arg

        # The ops that will populate the SMT function
        new_ops = list[Operation]()

        # When we enter the function, we are not in UB
        init_ub = ConstantBoolOp.from_bool(False)
        new_ops.append(init_ub)
        ub_value = init_ub.res

        ops_without_return = op.body.ops if not isinstance(
            op.body.ops[-1], Return) else op.body.ops[:-1]

        for body_op in ops_without_return:
            converted_ops, new_ub_value = convert_op(body_op, ssa_mapping)
            new_ops.extend(converted_ops)
            or_ubs = OrOp.get(ub_value, new_ub_value)
            new_ops.append(or_ubs)
            ub_value = or_ubs.res

        if isinstance((return_op := op.body.ops[-1]), Return):
            merge_ops, ret_value = merge_values_with_pairs(
                [ub_value] + [ssa_mapping[arg] for arg in return_op.arguments])
            new_ops.extend(merge_ops)
            new_ops.append(ReturnOp.from_ret_value(ret_value))
        else:
            new_ops.append(ReturnOp.from_ret_value(ub_value))

        rewriter.insert_op_at_pos(new_ops, smt_func.body.blocks[0], 0)
        rewriter.replace_matched_op(smt_func, new_results=[])


class ArithToSMT(ModulePass):
    name = "arith-to-smt"

    def apply(self, ctx: MLContext, op: ModuleOp):
        walker = PatternRewriteWalker(FuncToSMTPattern())
        walker.rewrite_module(op)
