from xdsl.ir import Attribute, MLContext, Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern, op_type_rewrite_pattern
from xdsl.dialects.builtin import IntegerType, ModuleOp, FunctionType
from xdsl.dialects.func import FuncOp, Return
from xdsl.printer import Printer

from dialects.smt_bitvector_dialect import BitVectorType, OrOp
from dialects.smt_printer_interface import SMTLibSort
from dialects.smt_utils_dialect import AnyPairType, PairOp, PairType
from dialects.smt_dialect import BoolType, ConstantBoolOp, DefineFunOp, ReturnOp


class FuncToSMTPattern(RewritePattern):
    """Convert func.func to an SMT formula"""

    def convert_type(self, type: Attribute) -> Attribute:
        """Convert a type to an SMT sort"""
        if isinstance(type, IntegerType):
            return PairType.from_params(
                BitVectorType.from_int(type.width.data), BoolType())
        if isinstance(type, SMTLibSort):
            return type
        raise Exception("Cannot convert {type} attribute")

    def convert_op(
        self, op: Operation,
        ssa_mapping: dict[SSAValue,
                          SSAValue]) -> tuple[list[Operation], SSAValue]:
        raise NotImplementedError()

    def merge_values_with_pairs(
            self, vals: list[SSAValue]) -> tuple[list[Operation], SSAValue]:
        """Merge a nonempty list of SSAValues into a single SSAValue (using smt.utils.pair)"""
        assert len(vals) > 0

        res: SSAValue = vals[-1]
        new_ops = list[Operation]()
        for i in reversed(range(len(vals) - 1)):
            new_op = PairOp.from_values(vals[i], res)
            new_ops.append(new_op)
            res = new_op.res
        return new_ops, res

    def merge_types_with_pairs(self, types: list[Attribute]) -> Attribute:
        """Merge a nonempty list of types into a single type (using smt.utils.pair)"""
        assert len(types) > 0

        res: Attribute = types[-1]
        for i in reversed(range(len(types) - 1)):
            res = AnyPairType.from_params(types[i], res)
        return res

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter):
        # We only handle single-block regions for now
        assert len(op.body.blocks) == 1

        # Mapping from old SSAValue to the new ones
        ssa_mapping = dict[SSAValue, SSAValue]()

        operand_types = [
            self.convert_type(input) for input in op.function_type.inputs.data
        ]
        result_types = self.merge_types_with_pairs([BoolType()] + [
            self.convert_type(output)
            for output in op.function_type.outputs.data
        ])

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
            converted_ops, new_ub_value = self.convert_op(body_op, ssa_mapping)
            new_ops.extend(converted_ops)
            or_ubs = OrOp.get(ub_value, new_ub_value)
            new_ops.append(or_ubs)
            ub_value = or_ubs.res

        if isinstance((return_op := op.body.ops[-1]), Return):
            merge_ops, ret_value = self.merge_values_with_pairs(
                [ub_value] + [ssa_mapping[arg] for arg in return_op.arguments])
            new_ops.extend(merge_ops)
            new_ops.append(ReturnOp.from_ret_value(ret_value))
        else:
            new_ops.append(ReturnOp.from_ret_value(ub_value))

        rewriter.insert_op_at_pos(new_ops, smt_func.body.blocks[0], 0)
        rewriter.replace_matched_op(smt_func, new_results=[])


def arith_to_smt(ctx: MLContext, module: ModuleOp):
    walker = PatternRewriteWalker(FuncToSMTPattern())
    walker.rewrite_module(module)
