from typing import Mapping, Sequence, cast
from abc import abstractmethod, ABC
from xdsl.ir import SSAValue, Attribute, Operation, Region, Block
from xdsl.pattern_rewriter import PatternRewriter
from xdsl_smt.dialects import smt_utils_dialect as smt_utils
from xdsl_smt.dialects import smt_int_dialect as smt_int
from xdsl_smt.dialects import smt_dialect as smt
from xdsl_smt.dialects import transfer
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.dialects.builtin import IntegerType, AnyIntegerAttr, IntegerAttr, FunctionType
from xdsl_smt.dialects.smt_dialect import BoolType


class IntAccessor(ABC):
    @abstractmethod
    def get_int_type(self, typ: Attribute) -> Attribute:
        ...

    @abstractmethod
    def get_payload(self, smt_integer: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        ...

    @abstractmethod
    def get_width(self, smt_integer: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        ...

    @abstractmethod
    def get_poison(self, smt_integer: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        ...

    @abstractmethod
    def get_int_max(
        self, smt_integer: SSAValue, typ: Attribute, rewriter: PatternRewriter
    ) -> SSAValue:
        ...

    @abstractmethod
    def get_packed_integer(
        self,
        payload: SSAValue,
        poison: SSAValue,
        width: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        ...


class FullIntAccessor(IntAccessor):
    def get_int_type(self, typ: Attribute) -> Attribute:
        inner_pair_type = smt_utils.PairType(smt_int.SMTIntType(), smt.BoolType())
        outer_pair_type = smt_utils.PairType(inner_pair_type, smt_int.SMTIntType())
        return outer_pair_type

    def get_payload(self, smt_integer: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        get_inner_pair_op = smt_utils.FirstOp(smt_integer)
        get_payload_op = smt_utils.FirstOp(get_inner_pair_op.res)
        rewriter.insert_op_before_matched_op([get_inner_pair_op, get_payload_op])
        return get_payload_op.res

    def get_width(self, smt_integer: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        get_width_op = smt_utils.SecondOp(smt_integer)
        rewriter.insert_op_before_matched_op([get_width_op])
        return get_width_op.res

    def get_poison(self, smt_integer: SSAValue, rewriter: PatternRewriter) -> SSAValue:
        get_inner_pair_op = smt_utils.FirstOp(smt_integer)
        get_poison_op = smt_utils.SecondOp(get_inner_pair_op.res)
        rewriter.insert_op_before_matched_op([get_inner_pair_op, get_poison_op])
        return get_poison_op.res

    def get_int_max(
        self, smt_integer: SSAValue, typ: Attribute, rewriter: PatternRewriter
    ) -> SSAValue:
        assert isinstance(typ, IntegerType)
        int_max_op = smt_int.ConstantOp(2**typ.width.data)
        rewriter.insert_op_before_matched_op([int_max_op])
        return int_max_op.res

    def get_packed_integer(
        self,
        payload: SSAValue,
        poison: SSAValue,
        width: SSAValue,
        rewriter: PatternRewriter,
    ) -> SSAValue:
        inner_pair = smt_utils.PairOp(payload, poison)
        outer_pair = smt_utils.PairOp(inner_pair.res, width)
        rewriter.insert_op_before_matched_op([inner_pair, outer_pair])
        return outer_pair.res


class PowEnabledIntAccessor(FullIntAccessor):
    def __init__(self):
        self.pow2_fun = None
        self.andi_fun = None
        self.xori_fun = None

    def _get_insert_point_above(self, var: SSAValue):
        owner = var.owner
        if isinstance(owner, Block):
            block = owner
        else:
            block = owner.parent
        assert isinstance(block, Block)
        insert_point = InsertPoint.at_start(block)
        return insert_point

    def andi(
        self, width: SSAValue, lhs: SSAValue, rhs: SSAValue, rewriter: PatternRewriter
    ):
        if self.andi_fun is None:
            pow2 = self.pow2_of
            insert_point = self._get_insert_point_above(lhs)
            pow2_fun = self.get_constraint_pow2(rewriter)
            # andi_fun_op = insert_bitwise(insert_point, integer_andi, "andi")
            andi_fun_op = insert_and_constraint_andi(rewriter, pow2_fun)
            self.andi_fun = andi_fun_op.ret
        result_op = smt.CallOp(self.andi_fun, [width, lhs, rhs])
        rewriter.insert_op_before_matched_op([result_op])
        return result_op.res[0]

    def xori(
        self, width: SSAValue, lhs: SSAValue, rhs: SSAValue, rewriter: PatternRewriter
    ):
        if self.xori_fun is None:
            insert_point = self._get_insert_point_above(lhs)
            andi_fun_op = insert_bitwise(insert_point, integer_xori, "xori")
            self.xori_fun = andi_fun_op.ret
        result_op = smt.CallOp(self.xori_fun, [width, lhs, rhs])
        rewriter.insert_op_before_matched_op([result_op])
        return result_op.res[0]

    def get_constraint_pow2(self, rewriter: PatternRewriter):
        if self.pow2_fun is None:
            pow2_fun_op = insert_pow2(rewriter)
            assert_op = constraint_pow2(rewriter, pow2_fun_op.ret)
            self.pow2_fun = pow2_fun_op.ret
        return self.pow2_fun

    def pow2_of(self, exp: SSAValue, rewriter: PatternRewriter):
        insert_point = self._get_insert_point_above(exp)
        pow2_fun = self.get_constraint_pow2(rewriter)
        result_op = smt.CallOp(pow2_fun, [exp])
        rewriter.insert_op_before_matched_op([result_op])
        return result_op.res[0]

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
            width = self.get_width(smt_integer, rewriter)
            int_max = self.pow2_of(width, rewriter)
        else:
            assert False
        return int_max


def insert_and_constraint_pow2(rewriter):
    declare_pow_op = insert_pow2(rewriter)
    assert_op = constraint_pow2(rewriter, declare_pow_op.ret)
    return declare_pow_op


def insert_pow2(rewriter):
    declare_pow_op = smt.DeclareFunOp(
        name="pow2",
        func_type=FunctionType.from_lists(
            inputs=[smt_int.SMTIntType()],
            outputs=[smt_int.SMTIntType()],
        ),
    )
    rewriter.insert_op_before_matched_op(declare_pow_op)
    return declare_pow_op


def constraint_pow2(rewriter, pow2_fun):
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
    rewriter.insert_op_before_matched_op([forall_op0, assert_op0])
    return assert_op0


def insert_and_constraint_andi(rewriter, pow2: SSAValue):
    declare_andi_op = smt.DeclareFunOp(
        name="andi",
        func_type=FunctionType.from_lists(
            inputs=[smt_int.SMTIntType(), smt_int.SMTIntType(), smt_int.SMTIntType()],
            outputs=[smt_int.SMTIntType()],
        ),
    )
    rewriter.insert_op_before_matched_op([declare_andi_op])
    # (declare-fun andi ((Int) (Int) (Int)) Int)
    # (assert (forall ((k Int) (kp Int) (x Int) (y Int))
    #                 (=>  (and (> kp k) (and (< x (pow2 k)) (< y (pow2 k))))
    #                     (= (andi k x y) (andi kp x y)))))
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
    # Left member of the implication
    kp_gt_k_op = smt_int.GtOp(kp, k)
    k_max_op = pow_x_op = smt.CallOp(pow2, [k])
    x_lt_k_max_op = smt_int.LtOp(x, k_max_op.res[0])
    y_lt_k_max_op = smt_int.LtOp(y, k_max_op.res[0])
    and0_op = smt.AndOp(kp_gt_k_op.res, x_lt_k_max_op.res)
    left_member_op = smt.AndOp(and0_op.res, y_lt_k_max_op.res)
    # Right member of the implication
    andi_k_x_y_op = smt.CallOp(declare_andi_op.ret, [k, x, y])
    andi_kp_x_y_op = smt.CallOp(declare_andi_op.ret, [kp, x, y])
    right_member_op = smt.EqOp(andi_k_x_y_op.res[0], andi_kp_x_y_op.res[0])
    #
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
    forall_op0 = smt.ForallOp.create(result_types=[BoolType()], regions=[body])
    assert_op0 = smt.AssertOp(forall_op0.res)
    rewriter.insert_op_before_matched_op([forall_op0, assert_op0])
    return declare_andi_op


def integer_andi(x_bit: SSAValue, y_bit: SSAValue) -> Sequence[Operation]:
    andi_op = smt_int.MulOp(x_bit, y_bit)
    return [andi_op]


def integer_xori(x_bit: SSAValue, y_bit: SSAValue) -> Sequence[Operation]:
    two_op = smt_int.ConstantOp(2)
    add_op = smt_int.AddOp(x_bit, y_bit)
    xori_op = smt_int.ModOp(add_op.res, two_op.res)
    return [two_op, add_op, xori_op]


def insert_bitwise(insert_point: InsertPoint, combine_bits, name):
    # Full implementation
    block = Block(
        arg_types=[smt_int.SMTIntType(), smt_int.SMTIntType(), smt_int.SMTIntType()]
    )
    k = block.args[0]
    x = block.args[1]
    y = block.args[2]
    # Constants
    one_op = smt_int.ConstantOp(1)
    two_op = smt_int.ConstantOp(2)
    block.add_ops([one_op, two_op])
    # Combine bits
    x_bit_op = smt_int.ModOp(x, two_op.res)
    y_bit_op = smt_int.ModOp(y, two_op.res)
    block.add_ops([x_bit_op, y_bit_op])
    bits_ops = combine_bits(x_bit_op.res, y_bit_op.res)
    assert bits_ops
    block.add_ops(bits_ops)
    # Recursive call
    new_x_op = smt_int.DivOp(x, two_op.res)
    new_y_op = smt_int.DivOp(y, two_op.res)
    k_minus_one = smt_int.SubOp(k, one_op.res)
    rec_call_op = smt.RecCallOp(
        args=[k_minus_one.res, new_x_op.res, new_y_op.res],
        result_types=[smt_int.SMTIntType()],
    )
    mul_op = smt_int.MulOp(two_op.res, rec_call_op.res[0])
    # Result
    res_op = smt_int.AddOp(bits_ops[-1].res, mul_op.res)
    return_op = smt.ReturnOp(res_op.res)
    # Build the function
    block.add_ops(
        [
            new_x_op,
            new_y_op,
            k_minus_one,
            rec_call_op,
            mul_op,
            res_op,
            return_op,
        ]
    )
    region = Region([block])
    define_fun_op = smt.DefineRecFunOp(region, name)
    Rewriter.insert_op(define_fun_op, insert_point)
    return define_fun_op
