from xdsl.dialects.builtin import ModuleOp

from .transfer_function_util import (
    replace_abstract_value_width,
    get_argument_widths_with_effect,
    get_argument_instances_with_effect,
    call_function_and_assert_result_with_effect,
    call_function_with_effect,
    get_result_instance_with_effect,
    insert_result_instances_to_block_with_effect,
    call_function_and_eq_result_with_effect,
    compress_and_op,
    insert_argument_instances_to_block_with_effect,
)
from ..dialects.smt_dialect import (
    DefineFunOp,
    AssertOp,
    CheckSatOp,
    ConstantBoolOp,
    ImpliesOp,
    YieldOp,
    ForallOp,
    AndOp,
    OrOp,
    XorOp,
    EqOp,
    IteOp,
)
from ..dialects.smt_bitvector_dialect import (
    ConstantOp,
)
from ..dialects.smt_utils_dialect import FirstOp
from xdsl.dialects.func import FuncOp
from ..dialects.transfer import AbstractValueType
from xdsl.ir import Operation, SSAValue, Attribute, Block, Region
from ..utils.transfer_function_util import (
    call_function_and_assert_result,
    get_result_width,
    SMTTransferFunction,
    FunctionCollection,
)


def valid_abstract_domain_check(
    transfer_function: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    int_attr: dict[int, int],
):
    """
    Given a transfer function, check if its result construct a valid abstract domain. For example,
    the upper bound of an integer range mush be larger or equal to its lower bound.
    """
    effect = ConstantBoolOp(False)
    abstract_func = transfer_function.transfer_function
    abs_op_constraint = transfer_function.abstract_constraint
    assert abstract_func is not None
    abs_arg_ops = get_argument_instances_with_effect(abstract_func, int_attr)
    abs_args: list[SSAValue] = [arg.res for arg in abs_arg_ops]
    is_abstract_arg = transfer_function.is_abstract_arg

    constant_bv_0 = ConstantOp(0, 1)
    constant_bv_1 = ConstantOp(1, 1)

    arg_widths = get_argument_widths_with_effect(abstract_func)
    result_width = get_result_width(abstract_func)

    abs_domain_constraints_ops: list[Operation] = []
    for i, abs_arg in enumerate(abs_args):
        if is_abstract_arg[i]:
            abs_domain_constraints_ops += call_function_and_assert_result_with_effect(
                domain_constraint.getFunctionByWidth(arg_widths[i]),
                [abs_arg],
                constant_bv_1,
                effect.res,
            )

    abs_arg_constraints_ops: list[Operation] = []
    if abs_op_constraint is not None:
        abs_arg_constraints_ops = call_function_and_assert_result_with_effect(
            abs_op_constraint, abs_args, constant_bv_1, effect.res
        )

    call_abs_func_op, call_abs_func_first_op = call_function_with_effect(
        abstract_func, abs_args, effect.res
    )
    abs_result_domain_invalid_ops = call_function_and_assert_result_with_effect(
        domain_constraint.getFunctionByWidth(result_width),
        [call_abs_func_first_op.res],
        constant_bv_0,
        effect.res,
    )
    return (
        [effect]
        + abs_arg_ops
        + [constant_bv_0, constant_bv_1]
        + abs_domain_constraints_ops
        + abs_arg_constraints_ops
        + [call_abs_func_op, call_abs_func_first_op]
        + abs_result_domain_invalid_ops
    )


def int_attr_check(
    transfer_function: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    int_attr: dict[int, int],
) -> list[Operation]:
    """
    Given the transfer function and a set of integer attributes associated with the function,
    returns a list of operations that checks if the operation is valid.
    For example:
    trunc %a from i32 to i64 (invalid)
    trunc %a from i5 to i3 (valid)
    """
    if transfer_function.int_attr_constraint is not None:
        effect = ConstantBoolOp(False)
        int_attr_constraint = transfer_function.int_attr_constraint
        int_attr_constraint_arg_ops = get_argument_instances_with_effect(
            int_attr_constraint, int_attr
        )
        int_attr_constraint_arg: list[SSAValue] = [
            arg.res for arg in int_attr_constraint_arg_ops
        ]

        constant_bv_1 = ConstantOp(1, 1)

        call_constraint_ops = call_function_and_assert_result_with_effect(
            int_attr_constraint, int_attr_constraint_arg, constant_bv_1, effect.res
        )
        return (
            [effect]
            + int_attr_constraint_arg_ops
            + [constant_bv_1]
            + call_constraint_ops
            + [CheckSatOp()]
        )
    else:
        true_op = ConstantBoolOp(True)
        assert_op = AssertOp(true_op.res)
        return [true_op, assert_op, CheckSatOp()]


def forward_soundness_check(
    transfer_function: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    int_attr: dict[int, int],
) -> list[Operation]:
    """
    Returns a list of operations that checks the soundness for a forward transfer function.
    This property checks for all concrete result, they must be included by the abstract value result
    produced by the transfer function.
    """
    assert transfer_function.is_forward
    abstract_func = transfer_function.transfer_function
    concrete_func = transfer_function.concrete_function
    abs_op_constraint = transfer_function.abstract_constraint
    op_constraint = transfer_function.op_constraint
    is_abstract_arg = transfer_function.is_abstract_arg

    assert abstract_func is not None
    assert concrete_func is not None

    abs_arg_ops = get_argument_instances_with_effect(abstract_func, int_attr)
    abs_args: list[SSAValue] = [arg.res for arg in abs_arg_ops]
    crt_arg_ops = get_argument_instances_with_effect(concrete_func, int_attr)
    crt_args_with_poison: list[SSAValue] = [arg.res for arg in crt_arg_ops]
    crt_arg_first_ops: list[FirstOp] = [FirstOp(arg) for arg in crt_args_with_poison]
    crt_args: list[SSAValue] = [arg.res for arg in crt_arg_first_ops]

    assert len(abs_args) == len(crt_args)
    arg_widths = get_argument_widths_with_effect(concrete_func)
    result_width = get_result_width(concrete_func)

    effect = ConstantBoolOp(False)
    constant_bv_0 = ConstantOp(0, 1)
    constant_bv_1 = ConstantOp(1, 1)

    abs_arg_include_crt_arg_constraints_ops: list[Operation] = []
    abs_domain_constraints_ops: list[Operation] = []
    for i, (abs_arg, crt_arg) in enumerate(zip(abs_args, crt_args)):
        if is_abstract_arg[i]:
            abs_arg_include_crt_arg_constraints_ops += (
                call_function_and_assert_result_with_effect(
                    instance_constraint.getFunctionByWidth(arg_widths[i]),
                    [abs_arg, crt_arg],
                    constant_bv_1,
                    effect.res,
                )
            )
            abs_domain_constraints_ops += call_function_and_assert_result_with_effect(
                domain_constraint.getFunctionByWidth(arg_widths[i]),
                [abs_arg],
                constant_bv_1,
                effect.res,
            )

    abs_arg_constraints_ops: list[Operation] = []
    if abs_op_constraint is not None:
        abs_arg_constraints_ops = call_function_and_assert_result_with_effect(
            abs_op_constraint, abs_args, constant_bv_1, effect.res
        )
    crt_args_constraints_ops: list[Operation] = []
    if op_constraint is not None:
        crt_args_constraints_ops = call_function_and_assert_result_with_effect(
            op_constraint, crt_args, constant_bv_1, effect.res
        )

    call_abs_func_op, call_abs_func_first_op = call_function_with_effect(
        abstract_func, abs_args, effect.res
    )
    call_crt_func_op, call_crt_func_first_op = call_function_with_effect(
        concrete_func, crt_args_with_poison, effect.res
    )
    call_crt_first_op = FirstOp(call_crt_func_first_op.res)

    abs_result_not_include_crt_result_ops = call_function_and_assert_result_with_effect(
        instance_constraint.getFunctionByWidth(result_width),
        [call_abs_func_first_op.res, call_crt_first_op.res],
        constant_bv_0,
        effect.res,
    )

    return (
        [effect]
        + abs_arg_ops
        + crt_arg_ops
        + crt_arg_first_ops
        + [constant_bv_0, constant_bv_1]
        + abs_domain_constraints_ops
        + abs_arg_include_crt_arg_constraints_ops
        + abs_arg_constraints_ops
        + crt_args_constraints_ops
        + [
            call_abs_func_op,
            call_abs_func_first_op,
            call_crt_func_op,
            call_crt_func_first_op,
            call_crt_first_op,
        ]
        + abs_result_not_include_crt_result_ops
        + [CheckSatOp()]
    )


def backward_soundness_check(
    transfer_function: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    int_attr: dict[int, int],
) -> list[Operation]:
    """
    Return a list of operations that checks the soundness for a backward transfer function. First, by applying
    the backwards transfer function, we can get the abstract domain of i-th operand. Next, we check if the abstract
    result includes all concrete result.
    """
    assert not transfer_function.is_forward
    operationNo = transfer_function.operationNo
    abstract_func = transfer_function.transfer_function
    concrete_func = transfer_function.concrete_function
    abs_op_constraint = transfer_function.abstract_constraint
    op_constraint = transfer_function.op_constraint
    is_abstract_arg = transfer_function.is_abstract_arg

    effect = ConstantBoolOp(False)
    assert abstract_func is not None
    assert concrete_func is not None
    arg_widths = get_argument_widths_with_effect(concrete_func)
    result_width = get_result_width(concrete_func)

    # replace the only abstract arg in transfer_function with bv with result_width
    assert sum(is_abstract_arg) == 1
    abs_arg_idx = is_abstract_arg.index(True)
    old_abs_arg = abstract_func.body.block.args[abs_arg_idx]
    assert isinstance(old_abs_arg.type, Attribute)
    new_abs_arg_type = replace_abstract_value_width(old_abs_arg.type, result_width)
    new_abs_arg = abstract_func.body.block.insert_arg(new_abs_arg_type, abs_arg_idx)
    abstract_func.body.block.args[abs_arg_idx + 1].replace_by(new_abs_arg)
    abstract_func.body.block.erase_arg(old_abs_arg)

    abs_arg_ops = get_argument_instances_with_effect(abstract_func, int_attr)
    abs_args: list[SSAValue] = [arg.res for arg in abs_arg_ops]

    crt_arg_ops = get_argument_instances_with_effect(concrete_func, int_attr)
    crt_args_with_poison: list[SSAValue] = [arg.res for arg in crt_arg_ops]
    crt_arg_first_ops = [FirstOp(arg) for arg in crt_args_with_poison]
    crt_args: list[SSAValue] = [arg.res for arg in crt_arg_first_ops]

    constant_bv_0 = ConstantOp(0, 1)
    constant_bv_1 = ConstantOp(1, 1)

    call_abs_func_op, call_abs_func_first_op = call_function_with_effect(
        abstract_func, abs_args, effect.res
    )
    call_crt_func_op, call_crt_func_first_op = call_function_with_effect(
        concrete_func, crt_args_with_poison, effect.res
    )
    call_crt_func_res_op = FirstOp(call_crt_func_first_op.res)

    abs_domain_constraints_ops = call_function_and_assert_result_with_effect(
        domain_constraint.getFunctionByWidth(result_width),
        [abs_args[0]],
        constant_bv_1,
        effect.res,
    )

    abs_arg_include_crt_res_constraint_ops = (
        call_function_and_assert_result_with_effect(
            instance_constraint.getFunctionByWidth(result_width),
            [abs_args[0], call_crt_func_res_op.res],
            constant_bv_1,
            effect.res,
        )
    )

    abs_arg_constraints_ops: list[Operation] = []
    if abs_op_constraint is not None:
        abs_arg_constraints_ops = call_function_and_assert_result(
            abs_op_constraint, abs_args, constant_bv_1
        )
    crt_args_constraints_ops: list[Operation] = []
    if op_constraint is not None:
        crt_args_constraints_ops = call_function_and_assert_result_with_effect(
            op_constraint, crt_args, constant_bv_1, effect.res
        )

    abs_result_not_include_crt_arg_constraint_ops = (
        call_function_and_assert_result_with_effect(
            instance_constraint.getFunctionByWidth(arg_widths[operationNo]),
            [call_abs_func_first_op.res, crt_args[operationNo]],
            constant_bv_0,
            effect.res,
        )
    )

    return (
        [effect]
        + abs_arg_ops
        + crt_arg_ops
        + crt_arg_first_ops
        + [constant_bv_0, constant_bv_1]
        + [
            call_abs_func_op,
            call_abs_func_first_op,
            call_crt_func_op,
            call_crt_func_first_op,
            call_crt_func_res_op,
        ]
        + abs_domain_constraints_ops
        + abs_arg_include_crt_res_constraint_ops
        + abs_arg_constraints_ops
        + crt_args_constraints_ops
        + abs_result_not_include_crt_arg_constraint_ops
        + [CheckSatOp()]
    )


def counterexample_check(
    counter_func: FuncOp,
    smt_counter_func: DefineFunOp,
    domain_constraint: FunctionCollection,
    int_attr: dict[int, int],
):
    """
    Return a list of operations that check if the transfer function breaks any other constraints. Such constraints
    are described by the parameter counter_func. For example, if one bit in demanded bits is zero, it must not change
    the operation result.
    """
    is_abstract_arg: list[bool] = [
        isinstance(arg, AbstractValueType) for arg in counter_func.args
    ]
    effect = ConstantBoolOp(False)
    arg_ops = get_argument_instances_with_effect(smt_counter_func, int_attr)
    args: list[SSAValue] = [arg.res for arg in arg_ops]
    arg_widths = get_argument_widths_with_effect(smt_counter_func)

    constant_bv_1 = ConstantOp(1, 1)

    abs_domain_constraints_ops: list[Operation] = []
    for i, arg in enumerate(args):
        if is_abstract_arg[i]:
            abs_domain_constraints_ops += call_function_and_assert_result_with_effect(
                domain_constraint.getFunctionByWidth(arg_widths[i]),
                [arg],
                constant_bv_1,
                effect.res,
            )
    call_counterexample_func_ops = call_function_and_assert_result_with_effect(
        smt_counter_func, args, constant_bv_1, effect.res
    )

    return (
        [effect]
        + arg_ops
        + [constant_bv_1]
        + abs_domain_constraints_ops
        + call_counterexample_func_ops
        + [CheckSatOp()]
    )


def get_forall_abs_res_prime_constraint(
    abs_args: list[SSAValue],
    abs_res_prime: SSAValue,
    concrete_func: DefineFunOp,
    op_constraint: DefineFunOp | None,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    int_attr: dict[int, int],
    is_abstract_arg: list[bool],
    arg_widths: list[int],
    result_width: int,
    constant_bv_1: ConstantOp,
    effect: ConstantBoolOp,
) -> list[Operation]:
    """
    ForAll([arg0inst, arg1inst],
            Implies(And(getInstanceConstraint(arg0inst, arg0field0, arg0field1),
                        getInstanceConstraint(arg1inst, arg1field0, arg1field1)),
                    getInstanceConstraint(concrete_op(arg0inst, arg1inst), abs_res'0, abs_res'1)))
    This constraint is called abs_res_prime constraint
    """
    forall_abs_res_prime_constraint_block = Block()
    crt_args_with_poison: list[
        SSAValue
    ] = insert_argument_instances_to_block_with_effect(
        concrete_func, int_attr, forall_abs_res_prime_constraint_block
    )
    crt_arg_first_ops: list[FirstOp] = [FirstOp(arg) for arg in crt_args_with_poison]
    crt_args: list[SSAValue] = [arg.res for arg in crt_arg_first_ops]
    assert len(abs_args) == len(crt_args)

    abs_arg_include_crt_arg_constraints_ops: list[Operation] = []
    forall_abs_res_prime_antecedent_ops: list[Operation] = []
    for i, (abs_arg, crt_arg) in enumerate(zip(abs_args, crt_args)):
        if is_abstract_arg[i]:
            (
                abs_arg_include_crt_arg_call_tmp,
                abs_arg_include_crt_arg_call_tmp_eq,
            ) = call_function_and_eq_result_with_effect(
                instance_constraint.getFunctionByWidth(arg_widths[i]),
                [abs_arg, crt_arg],
                constant_bv_1,
                effect.res,
            )

            abs_arg_include_crt_arg_constraints_ops += abs_arg_include_crt_arg_call_tmp

            forall_abs_res_prime_antecedent_ops.append(
                abs_arg_include_crt_arg_call_tmp_eq
            )

    op_constraint_ops: list[Operation] = []
    if op_constraint is not None:
        (
            op_constraint_ops,
            op_constraint_eq_op,
        ) = call_function_and_eq_result_with_effect(
            op_constraint, crt_args, constant_bv_1, effect.res
        )
        forall_abs_res_prime_antecedent_ops.append(op_constraint_eq_op)
    (
        forall_abs_res_prime_antecedent_res,
        forall_abs_res_prime_antecedent_and_ops,
    ) = compress_and_op(forall_abs_res_prime_antecedent_ops)

    # crt_res_fist_op is with poison, so we need another first op
    crt_res_op, crt_res_first_op = call_function_with_effect(
        concrete_func, crt_args_with_poison, effect.res
    )
    crt_res_first_first_op = FirstOp(crt_res_first_op.res)

    (
        abs_res_prime_include_crt_res_ops,
        forall_abs_res_prime_consequent_eq,
    ) = call_function_and_eq_result_with_effect(
        instance_constraint.getFunctionByWidth(result_width),
        [abs_res_prime, crt_res_first_first_op.res],
        constant_bv_1,
        effect.res,
    )

    forall_abs_res_prime_imply_op = ImpliesOp(
        forall_abs_res_prime_antecedent_res, forall_abs_res_prime_consequent_eq.res
    )
    forall_abs_res_prime_yield_op = YieldOp(forall_abs_res_prime_imply_op)

    forall_abs_res_prime_constraint_block.add_ops(
        crt_arg_first_ops
        + abs_arg_include_crt_arg_constraints_ops
        + op_constraint_ops
        + forall_abs_res_prime_antecedent_and_ops
        + [crt_res_op, crt_res_first_op, crt_res_first_first_op]
        + abs_res_prime_include_crt_res_ops
        + [forall_abs_res_prime_imply_op, forall_abs_res_prime_yield_op]
    )

    forall_abs_res_prime_constraint = ForallOp.from_variables(
        [], Region(forall_abs_res_prime_constraint_block)
    )
    forall_abs_res_prime_constraint_assert = AssertOp(
        forall_abs_res_prime_constraint.res
    )

    forall_abs_res_prime_constraint_ops: list[Operation] = [
        forall_abs_res_prime_constraint,
        forall_abs_res_prime_constraint_assert,
    ]
    return forall_abs_res_prime_constraint_ops


def get_forall_crt_res_prime_constraint(
    abs_res: SSAValue,
    abs_res_prime: SSAValue,
    concrete_func: DefineFunOp,
    result_width: int,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    constant_bv_1: ConstantOp,
    effect: ConstantBoolOp,
) -> list[Operation]:
    """
    ForAll([crt_res_prime],
            Implies(getInstanceConstraint(crt_res_prime, abs_res_prime0, abs_res_prime0),
                    getInstanceConstraint(crt_res_prime, abs_res[0], abs_res[1])))
    This is we called forall_crt_res_prime constraint
    """
    forall_crt_res_prime_constraint_block = Block()
    crt_res_prime_with_poison = insert_result_instances_to_block_with_effect(
        concrete_func, forall_crt_res_prime_constraint_block
    )

    # crt_res_prime is with poison, thus we need do another firstOp here
    assert len(crt_res_prime_with_poison) == 1
    crt_res_prime_first_op = FirstOp(crt_res_prime_with_poison[0])
    crt_res_prime = crt_res_prime_first_op.res

    (
        abs_res_prime_include_crt_res_prime_ops,
        forall_crt_res_prime_constraint_antecedent_eq,
    ) = call_function_and_eq_result_with_effect(
        instance_constraint.getFunctionByWidth(result_width),
        [abs_res_prime, crt_res_prime],
        constant_bv_1,
        effect.res,
    )

    (
        abs_res_include_crt_res_prime_ops,
        forall_crt_res_prime_constraint_consequent_eq,
    ) = call_function_and_eq_result_with_effect(
        instance_constraint.getFunctionByWidth(result_width),
        [abs_res, crt_res_prime],
        constant_bv_1,
        effect.res,
    )

    forall_crt_res_prime_constraint_imply = ImpliesOp(
        forall_crt_res_prime_constraint_antecedent_eq.res,
        forall_crt_res_prime_constraint_consequent_eq.res,
    )
    forall_crt_res_prime_constraint_yield = YieldOp(
        forall_crt_res_prime_constraint_imply.res
    )

    forall_crt_res_prime_constraint_block.add_ops(
        [crt_res_prime_first_op]
        + abs_res_prime_include_crt_res_prime_ops
        + abs_res_include_crt_res_prime_ops
        + [forall_crt_res_prime_constraint_imply, forall_crt_res_prime_constraint_yield]
    )

    forall_crt_res_prime_constraint = ForallOp.from_variables(
        [], Region(forall_crt_res_prime_constraint_block)
    )
    forall_crt_res_prime_constraint_assert = AssertOp(
        forall_crt_res_prime_constraint.res
    )

    forall_crt_res_prime_constraint_ops: list[Operation] = [
        forall_crt_res_prime_constraint,
        forall_crt_res_prime_constraint_assert,
    ]
    return forall_crt_res_prime_constraint_ops


def forward_precision_check(
    transfer_function: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    int_attr: dict[int, int],
):
    assert transfer_function.is_forward
    abstract_func = transfer_function.transfer_function
    concrete_func = transfer_function.concrete_function
    abs_op_constraint = transfer_function.abstract_constraint
    op_constraint = transfer_function.op_constraint
    is_abstract_arg = transfer_function.is_abstract_arg

    assert abstract_func is not None
    assert concrete_func is not None

    abs_arg_ops = get_argument_instances_with_effect(abstract_func, int_attr)
    abs_args: list[SSAValue] = [arg.res for arg in abs_arg_ops]

    arg_widths = get_argument_widths_with_effect(concrete_func)
    result_width = get_result_width(concrete_func)

    effect = ConstantBoolOp(False)
    constant_bv_0 = ConstantOp(0, 1)
    constant_bv_1 = ConstantOp(1, 1)

    abs_domain_constraints_ops: list[Operation] = []
    for i, abs_arg in enumerate(abs_args):
        if is_abstract_arg[i]:
            abs_domain_constraints_ops += call_function_and_assert_result_with_effect(
                domain_constraint.getFunctionByWidth(arg_widths[i]),
                [abs_arg],
                constant_bv_1,
                effect.res,
            )

    abs_arg_constraints_ops: list[Operation] = []
    if abs_op_constraint is not None:
        abs_arg_constraints_ops = call_function_and_assert_result_with_effect(
            abs_op_constraint, abs_args, constant_bv_1, effect.res
        )

    abs_res_prime_ops, abs_res_prime = get_result_instance_with_effect(abstract_func)

    abs_res_prime_domain_constraints_ops = call_function_and_assert_result_with_effect(
        domain_constraint.getFunctionByWidth(result_width),
        [abs_res_prime],
        constant_bv_1,
        effect.res,
    )

    forall_abs_res_prime_constraint_ops = get_forall_abs_res_prime_constraint(
        abs_args,
        abs_res_prime,
        concrete_func,
        op_constraint,
        domain_constraint,
        instance_constraint,
        int_attr,
        is_abstract_arg,
        arg_widths,
        result_width,
        constant_bv_1,
        effect,
    )

    call_abs_func_op, call_abs_func_first_op = call_function_with_effect(
        abstract_func, abs_args, effect.res
    )

    forall_crt_res_prime_constraint_ops = get_forall_crt_res_prime_constraint(
        call_abs_func_first_op.res,
        abs_res_prime,
        concrete_func,
        result_width,
        domain_constraint,
        instance_constraint,
        constant_bv_1,
        effect,
    )

    # And(Not(getInstanceConstraint(abs_resInst, abs_res_prime0, abs_res_prime1)), getInstanceConstraint(abs_resInst, abs_res[0], abs_res[1]))

    abs_res_ele_ops, abs_res_ele_with_poison = get_result_instance_with_effect(
        concrete_func
    )
    abs_res_ele_first_op = FirstOp(abs_res_ele_with_poison)
    abs_res_ele = abs_res_ele_first_op.res

    abs_res_prime_not_include_abs_res_ele_ops = (
        call_function_and_assert_result_with_effect(
            instance_constraint.getFunctionByWidth(result_width),
            [abs_res_prime, abs_res_ele],
            constant_bv_0,
            effect.res,
        )
    )

    abs_res_include_abs_res_ele_ops = call_function_and_assert_result_with_effect(
        instance_constraint.getFunctionByWidth(result_width),
        [call_abs_func_first_op.res, abs_res_ele],
        constant_bv_1,
        effect.res,
    )

    return (
        [effect]
        + abs_arg_ops
        + [constant_bv_0, constant_bv_1]
        + abs_domain_constraints_ops
        + abs_arg_constraints_ops
        + abs_res_prime_ops
        + abs_res_prime_domain_constraints_ops
        + forall_abs_res_prime_constraint_ops
        + [call_abs_func_op, call_abs_func_first_op]
        + forall_crt_res_prime_constraint_ops
        + abs_res_ele_ops
        + [abs_res_ele_first_op]
        + abs_res_prime_not_include_abs_res_ele_ops
        + abs_res_include_abs_res_ele_ops
        + [CheckSatOp()]
    )


def backward_precision_check(
    transfer_function: SMTTransferFunction,
    domain_constraint: FunctionCollection,
    instance_constraint: FunctionCollection,
    int_attr: dict[int, int],
) -> list[Operation]:
    assert not transfer_function.is_forward
    return [CheckSatOp()]


def print_module_until_op(module: ModuleOp, cur_op: Operation):
    for op in module.body.block.ops:
        if op == cur_op:
            break
        print(op)


def module_op_validity_check(module: ModuleOp) -> bool:
    """
    This function checks if all op in module follows the dominance property
    """
    dom_vals: list[set[SSAValue]] = [set()]

    def find_in_dom_vals(val: SSAValue) -> bool:
        for dom_val in dom_vals:
            if val in dom_val:
                return True
        return False

    def op_type_check(op: Operation) -> bool:
        if (
            isinstance(op, AndOp)
            or isinstance(op, OrOp)
            or isinstance(op, XorOp)
            or isinstance(op, EqOp)
        ):
            return op.operands[0].type == op.operands[1].type
        elif isinstance(op, IteOp):
            return op.operands[1].type == op.operands[2].type
        return True

    for i, op in enumerate(module.ops):
        if isinstance(op, ForallOp):
            block_args: set[SSAValue] = set()
            block_vals: set[SSAValue] = set()
            dom_vals.append(block_args)
            dom_vals.append(block_vals)
            for arg in op.body.block.args:
                block_args.add(arg)
            for inside_op in op.body.block.ops:
                if not op_type_check(inside_op):
                    print(inside_op)
                    return False
                for val in inside_op.operands:
                    if not find_in_dom_vals(val):
                        print(val)
                        print(inside_op)
                        return False
                for res in inside_op.results:
                    block_vals.add(res)
            dom_vals.pop()
            dom_vals.pop()
            for res in op.results:
                dom_vals[-1].add(res)

        else:
            if not op_type_check(op):
                print(op)
                return False
            for val in op.operands:
                if not find_in_dom_vals(val):
                    print_module_until_op(module, op)
                    print(i, op)
                    return False
            for res in op.results:
                dom_vals[-1].add(res)

    return True
