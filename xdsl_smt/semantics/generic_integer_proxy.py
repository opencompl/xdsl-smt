"""
This files contains utilities to manipulate integer-based semantics for the
arithmetic dialect.
"""

from xdsl.ir import SSAValue, Attribute, Region, Block
from dataclasses import dataclass
from typing import Tuple
from xdsl.pattern_rewriter import PatternRewriter
from xdsl_smt.dialects import smt_utils_dialect as smt_utils
from xdsl_smt.dialects import smt_int_dialect as smt_int
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import transfer
from xdsl.rewriter import InsertPoint
from xdsl.dialects.builtin import IntegerType, FunctionType
from xdsl_smt.dialects.smt_dialect import BoolType


@dataclass
class IntegerProxy:
    """
    Provide utilities to manipulate integer-based semantics.
    In particular, contains references to the pow2 and andi functions, if they
    are already defined in the IR.
    """

    pow2_fun: SSAValue | None = None
    andi_fun: SSAValue | None = None

    def _get_insert_point_above(self, var: SSAValue):
        owner = var.owner
        if isinstance(owner, Block):
            block = owner
        else:
            block = owner.parent
        assert isinstance(block, Block)
        insert_point = InsertPoint.at_start(block)
        return insert_point

    @property
    def int_type(self) -> Attribute:
        """Get the representation of an integer in the generic semantics
        framework."""
        inner_pair_type = smt_utils.PairType(smt_int.SMTIntType(), smt.BoolType())
        outer_pair_type = smt_utils.PairType(inner_pair_type, smt_int.SMTIntType())
        return outer_pair_type

    def unpack_integer(
        self, smt_integer: SSAValue, rewriter: PatternRewriter
    ) -> Tuple[SSAValue, SSAValue, SSAValue]:
        "Extract the different pieces of a generic integer."
        payload = self._get_payload(smt_integer, rewriter)
        poison = self._get_poison(smt_integer, rewriter)
        width = self._get_width(smt_integer, rewriter)
        return payload, poison, width

    def _get_payload(
        self, smt_integer: SSAValue, rewriter: PatternRewriter
    ) -> SSAValue:
        get_inner_pair_op = smt_utils.FirstOp(smt_integer)
        get_payload_op = smt_utils.FirstOp(get_inner_pair_op.res)
        rewriter.insert_op_before_matched_op([get_inner_pair_op, get_payload_op])
        return get_payload_op.res

    def _get_width(self, smt_integer: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        get_width_op = smt_utils.SecondOp(smt_integer)
        rewriter.insert_op_before_matched_op([get_width_op])
        return get_width_op.res

    def _get_poison(self, smt_integer: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        get_inner_pair_op = smt_utils.FirstOp(smt_integer)
        get_poison_op = smt_utils.SecondOp(get_inner_pair_op.res)
        rewriter.insert_op_before_matched_op([get_inner_pair_op, get_poison_op])
        return get_poison_op.res

    def pack_integer(
        self,
        payload: SSAValue,
        poison: SSAValue,
        width: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        "Build a generic integer starting from its different pieces."
        mod_payload = self.get_signed_to_unsigned(payload, width, rewriter)
        inner_pair = smt_utils.PairOp(mod_payload, poison)
        outer_pair = smt_utils.PairOp(inner_pair.res, width)
        rewriter.insert_op_before_matched_op([inner_pair, outer_pair])
        return outer_pair.res

    def andi_of(
        self, width: SSAValue, lhs: SSAValue, rhs: SSAValue, rewriter: PatternRewriter
    ) -> SSAValue:
        "Compute the bitwise And of the parameters."
        if self.andi_fun is None:
            pow2_fun = self.get_pow2(rewriter)
            andi_fun_op = insert_and_constraint_andi(rewriter, pow2_fun)
            self.andi_fun = andi_fun_op.ret
        result_op = smt.CallOp(self.andi_fun, [width, lhs, rhs])
        rewriter.insert_op_before_matched_op([result_op])
        return result_op.res[0]

    def pow2_of(self, exp: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        "Compute the pow2 of the parameter."
        pow2_fun = self.get_pow2(rewriter)
        result_op = smt.CallOp(pow2_fun, [exp])
        rewriter.insert_op_before_matched_op([result_op])
        return result_op.res[0]

    def build_pow2(self, rewriter: PatternRewriter):
        if self.pow2_fun is None:
            pow2_fun_op = insert_and_constraint_pow2(rewriter)
            self.pow2_fun = pow2_fun_op.ret
        assert self.pow2_fun

    def get_pow2(self, rewriter: PatternRewriter) -> SSAValue:
        self.build_pow2(rewriter)
        assert self.pow2_fun
        return self.pow2_fun

    def get_int_max(
        self, smt_integer: SSAValue, typ: Attribute, rewriter: PatternRewriter
    ) -> SSAValue:
        if isinstance(typ, IntegerType):
            int_max_op = smt_int.ConstantOp(2**typ.width.data)
            rewriter.insert_op_before_matched_op([int_max_op])
            int_max = int_max_op.res
        elif isinstance(typ, transfer.TransIntegerType) or isinstance(
            typ, smt_int.SMTIntType
        ):
            width = self._get_width(smt_integer, rewriter)
            int_max = self.pow2_of(width, rewriter)
        else:
            assert False
        return int_max

    def get_signed_to_unsigned(
        self, payload: SSAValue, width: SSAValue, rewriter: PatternRewriter
    ) -> SSAValue:
        # (x + pow2(k)) mod pow2(k)
        int_max = self.pow2_of(width, rewriter)
        payload_norm_op = smt_int.AddOp(payload, int_max)
        payload_modulo_op = smt_int.ModOp(payload_norm_op.res, int_max)
        rewriter.insert_op_before_matched_op([payload_norm_op, payload_modulo_op])
        return payload_modulo_op.res

    def get_unsigned_to_signed(
        self, payload: SSAValue, width: SSAValue, rewriter: PatternRewriter
    ) -> SSAValue:
        # 2 · (x mod pow2(k − 1)) − x
        one_op = smt_int.ConstantOp(1)
        width_minus_one = smt_int.SubOp(width, one_op.res)
        rewriter.insert_op_before_matched_op([one_op, width_minus_one])
        int_max = self.pow2_of(width_minus_one.res, rewriter)
        modulo_op = smt_int.ModOp(payload, int_max)
        modulo_minus_payload_op = smt_int.SubOp(modulo_op.res, payload)
        two_op = smt_int.ConstantOp(2)
        times_2_op = smt_int.MulOp(modulo_minus_payload_op.res, two_op.res)
        rewriter.insert_op_before_matched_op(
            [modulo_op, modulo_minus_payload_op, two_op, times_2_op]
        )
        return times_2_op.res


def insert_and_constraint_pow2(rewriter: PatternRewriter) -> smt.DeclareFunOp:
    """Insert an uninterpreted function mimicking pow2
    and constraint the latter with assertions and quantifiers
    in order to help the solvers."""
    declare_pow_op = smt.DeclareFunOp(
        name="pow2",
        func_type=FunctionType.from_lists(
            inputs=[smt_int.SMTIntType()],
            outputs=[smt_int.SMTIntType()],
        ),
    )
    pow2_fun = declare_pow_op.ret
    # Forall x,y, x > y => pow2(x) > pow2(y)
    body = Region([Block(arg_types=[smt_int.SMTIntType(), smt_int.SMTIntType()])])
    assert isinstance(body.first_block, Block)
    x = body.first_block.args[0]
    y = body.first_block.args[1]
    x_gt_y_op = smt_int.GtOp(x, y)
    pow_x_op = smt.CallOp(pow2_fun, [x])
    pow_y_op = smt.CallOp(pow2_fun, [y])
    pow_x_gt_y_op = smt_int.GtOp(pow_x_op.res[0], pow_y_op.res[0])
    implies_op = smt.ImpliesOp(
        x_gt_y_op.res,
        pow_x_gt_y_op.res,
    )
    yield_op = smt.YieldOp(implies_op.res)
    body.first_block.add_ops(
        [x_gt_y_op, pow_x_op, pow_y_op, pow_x_gt_y_op, implies_op, yield_op]
    )
    forall_op0 = smt.ForallOp.create(result_types=[BoolType()], regions=[body])
    assert_op0 = smt.AssertOp(forall_op0.res)
    rewriter.insert_op_before_matched_op([declare_pow_op, forall_op0, assert_op0])

    return declare_pow_op


def insert_and_constraint_andi(rewriter: PatternRewriter, pow2: SSAValue):
    """Insert an uninterpreted function mimicking the bitwise
    And, and constraint the latter with assertions and quantifiers
    in order to help the solvers."""
    # Declare
    declare_andi_op = smt.DeclareFunOp(
        name="andi",
        func_type=FunctionType.from_lists(
            inputs=[smt_int.SMTIntType(), smt_int.SMTIntType(), smt_int.SMTIntType()],
            outputs=[smt_int.SMTIntType()],
        ),
    )
    rewriter.insert_op_before_matched_op([declare_andi_op])
    # Forall region and parameters
    body = Region(
        [
            Block(
                arg_types=[
                    smt_int.SMTIntType(),
                    smt_int.SMTIntType(),
                    smt_int.SMTIntType(),
                    smt_int.SMTIntType(),
                ]
            )
        ]
    )
    assert isinstance(body.first_block, Block)
    k = body.first_block.args[0]
    kp = body.first_block.args[1]
    x = body.first_block.args[2]
    y = body.first_block.args[3]
    # Left member of the implication: (and (> kp k) (and (< x (pow2 k)) (< y (pow2 k))))
    kp_gt_k_op = smt_int.GtOp(kp, k)
    k_max_op = smt.CallOp(pow2, [k])
    x_lt_k_max_op = smt_int.LtOp(x, k_max_op.res[0])
    y_lt_k_max_op = smt_int.LtOp(y, k_max_op.res[0])
    and0_op = smt.AndOp(kp_gt_k_op.res, x_lt_k_max_op.res)
    left_member_op = smt.AndOp(and0_op.res, y_lt_k_max_op.res)
    # Right member of the implication: (= (andi k x y) (andi kp x y))
    andi_k_x_y_op = smt.CallOp(declare_andi_op.ret, [k, x, y])
    andi_kp_x_y_op = smt.CallOp(declare_andi_op.ret, [kp, x, y])
    right_member_op = smt.EqOp(andi_k_x_y_op.res[0], andi_kp_x_y_op.res[0])
    # Build the implication: (=> left right)
    implies_op = smt.ImpliesOp(
        left_member_op.res,
        right_member_op.res,
    )
    yield_op = smt.YieldOp(implies_op.res)
    body.first_block.add_ops(
        [
            kp_gt_k_op,
            k_max_op,
            x_lt_k_max_op,
            y_lt_k_max_op,
            and0_op,
            left_member_op,
            andi_k_x_y_op,
            andi_kp_x_y_op,
            right_member_op,
            implies_op,
            yield_op,
        ]
    )
    # Build the forall and the corresponding assertion
    forall_op0 = smt.ForallOp.create(result_types=[BoolType()], regions=[body])
    assert_op0 = smt.AssertOp(forall_op0.res)

    rewriter.insert_op_before_matched_op([forall_op0, assert_op0])
    return declare_andi_op
