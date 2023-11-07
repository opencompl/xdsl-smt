#!/usr/bin/env python3

import argparse
import subprocess

from xdsl.ir import MLContext, Operation
from xdsl.parser import Parser

from io import StringIO
from ..dialects.smt_dialect import (
    SMTDialect,
    DefineFunOp,
    DeclareConstOp,
    CallOp,
    AssertOp,
    CheckSatOp,
    EqOp,
)
from ..dialects.smt_bitvector_dialect import SMTBitVectorDialect, ConstantOp
from ..dialects.smt_utils_dialect import FirstOp
from ..dialects.arith_dialect import Arith
from ..dialects.index_dialect import Index
from ..dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func
from ..dialects.transfer import Transfer, AbstractValueType

from ..passes import calculate_smt as cs
from ..utils.trans_interpreter_smt import *
from ..passes.rename_values import RenameValuesPass
from ..passes.lower_to_smt.lower_to_smt import LowerToSMT, integer_poison_type_lowerer
from ..passes.pdl_to_smt import PDLToSMT
from ..passes.lower_to_smt.transfer_to_smt import (
    abstract_value_type_lowerer,
    transfer_integer_type_lowerer,
)
from ..passes.lower_to_smt import (
    arith_to_smt_patterns,
    func_to_smt_patterns,
    transfer_to_smt_patterns,
)
from ..traits.smt_printer import print_to_smtlib

from z3 import BitVec, Solver, And, Not, simplify, ForAll, Implies
import sys as sys


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "transfer_functions", type=str, nargs="?", help="path to the transfer functions"
    )


def parse_file(ctx, file: str | None) -> Operation:
    if file is None:
        f = sys.stdin
    else:
        f = open(file)

    parser = Parser(ctx, f.read(), file)
    module = parser.parse_op()
    return module


KEY_NEED_VERIFY = "builtin.NEED_VERIFY"
MAXIMAL_VERIFIED_BITS = 8


def solveVectorWidth():
    return list(range(1, 9))


def verify_pattern(ctx: MLContext, op: ModuleOp) -> bool:
    cloned_op = op.clone()
    PDLToSMT().apply(ctx, cloned_op)
    stream = StringIO()
    print_to_smtlib(cloned_op, stream)

    res = subprocess.run(
        ["z3", "-in"],
        capture_output=True,
        input=stream.getvalue(),
        text=True,
    )
    if res.returncode != 0:
        raise Exception(res.stderr)

    return "unsat" in res.stdout


def basic_constraint_check(abstract_func: DefineFunOp, get_constraint: DefineFunOp):
    abstract_type = get_constraint.body.block.args[0].type
    arg_constant: list[DeclareConstOp] = []
    arg_constraints: list[CallOp] = []
    arg_constraints_first: list[FirstOp] = []
    for arg in abstract_func.body.block.args:
        arg_constant.append(DeclareConstOp(arg.type))
        if arg.type == abstract_type:
            arg_constraints.append(
                CallOp.get(get_constraint.results[0], [arg_constant[-1].results[0]])
            )
            arg_constraints_first.append(FirstOp(arg_constraints[-1].results[0]))
    assert len(arg_constant) != 0

    abstract_result = CallOp.get(
        abstract_func.results[0], [op.results[0] for op in arg_constant]
    )
    result_constraint = CallOp.get(
        get_constraint.results[0], [abstract_result.results[0]]
    )
    result_constraint_first = FirstOp(result_constraint.results[0])

    constant_bv_0 = ConstantOp(0, 1)
    constant_bv_1 = ConstantOp(1, 1)

    eq_ops: list[EqOp] = []
    assert_ops: list[AssertOp] = []

    for c in arg_constraints_first:
        eq_ops.append(EqOp(constant_bv_1.results[0], c.results[0]))
        assert_ops.append(AssertOp(eq_ops[-1].results[0]))

    eq_ops.append(EqOp(constant_bv_0.results[0], result_constraint_first.results[0]))
    assert_ops.append(AssertOp(eq_ops[-1].results[0]))

    return (
        arg_constant
        + arg_constraints
        + arg_constraints_first
        + [abstract_result, result_constraint, result_constraint_first]
        + [constant_bv_1, constant_bv_0]
        + eq_ops
        + assert_ops
        + [CheckSatOp()]
    )


def soundness_check(
    abstract_func: DefineFunOp,
    concrete_func: DefineFunOp,
    get_constraint: DefineFunOp,
    get_inst_constraint: DefineFunOp,
):
    abstract_type = get_constraint.body.block.args[0].type
    instance_type = concrete_func.body.block.args[1].type
    arg_constant: list[DeclareConstOp] = []
    inst_constant: list[DeclareConstOp] = []
    arg_constraints: list[CallOp] = []
    inst_constraints: list[CallOp] = []
    arg_constraints_first: list[FirstOp] = []
    inst_constraints_first: list[FirstOp] = []

    for arg in abstract_func.body.block.args:
        arg_constant.append(DeclareConstOp(arg.type))
        if arg.type == abstract_type:
            arg_constraints.append(
                CallOp.get(get_constraint.results[0], [arg_constant[-1].results[0]])
            )
            arg_constraints_first.append(FirstOp(arg_constraints[-1].results[0]))

            inst_constant.append(DeclareConstOp(instance_type))
            inst_constraints.append(
                CallOp.get(
                    get_inst_constraint.results[0],
                    [arg_constant[-1].results[0], inst_constant[-1].results[0]],
                )
            )
            inst_constraints_first.append(FirstOp(inst_constraints[-1].results[0]))
        else:
            inst_constant.append(arg_constant[-1])

    assert len(arg_constant) != 0

    abstract_result = CallOp.get(
        abstract_func.results[0], [op.results[0] for op in arg_constant]
    )
    inst_result = CallOp.get(
        concrete_func.results[0], [op.results[0] for op in inst_constant]
    )
    inst_result_constraint = CallOp.get(
        get_inst_constraint.results[0],
        [abstract_result.results[0], inst_result.results[0]],
    )
    inst_result_constraint_first = FirstOp(inst_result_constraint.results[0])

    constant_bv_0 = ConstantOp(0, 1)
    constant_bv_1 = ConstantOp(1, 1)

    eq_ops: list[EqOp] = []
    assert_ops: list[AssertOp] = []

    for c in arg_constraints_first:
        eq_ops.append(EqOp(constant_bv_1.results[0], c.results[0]))
        assert_ops.append(AssertOp(eq_ops[-1].results[0]))

    for c in inst_constraints_first:
        eq_ops.append(EqOp(constant_bv_1.results[0], c.results[0]))
        assert_ops.append(AssertOp(eq_ops[-1].results[0]))
    eq_ops.append(
        EqOp(constant_bv_0.results[0], inst_result_constraint_first.results[0])
    )
    assert_ops.append(AssertOp(eq_ops[-1].results[0]))

    return (
        arg_constant
        + inst_constant
        + arg_constraints
        + inst_constraints
        + arg_constraints_first
        + inst_constraints_first
        + [
            abstract_result,
            inst_result,
            inst_result_constraint,
            inst_result_constraint_first,
        ]
        + [constant_bv_1, constant_bv_0]
        + eq_ops
        + assert_ops
        + [CheckSatOp()]
    )


def main() -> None:
    ctx = MLContext()
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    # Register all dialects
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)
    ctx.register_dialect(SMTDialect)
    ctx.register_dialect(SMTBitVectorDialect)
    ctx.register_dialect(SMTUtilsDialect)
    ctx.register_dialect(Transfer)
    ctx.register_dialect(Index)

    # Parse the files
    module = parse_file(ctx, args.transfer_functions)
    assert isinstance(module, ModuleOp)

    get_constraint = None
    get_instance_constraint = None

    func_name_to_func = {}

    RenameValuesPass().apply(ctx, module)
    for width in solveVectorWidth():
        print("Current width: ", width)
        smt_module: ModuleOp = module.clone()
        LowerToSMT.rewrite_patterns = [
            *arith_to_smt_patterns,
            *transfer_to_smt_patterns,
            *func_to_smt_patterns,
        ]
        LowerToSMT.type_lowerers = [
            integer_poison_type_lowerer,
            abstract_value_type_lowerer,
            lambda type: transfer_integer_type_lowerer(type, width),
        ]
        PDLToSMT().apply(ctx, smt_module)
        for func in smt_module.ops:
            if isinstance(func, DefineFunOp):
                func_name_to_func[func.fun_name.data] = func
                if func.fun_name.data == "getConstraint":
                    get_constraint = func
                elif func.fun_name.data == "getInstanceConstraint":
                    get_instance_constraint = func
        for func_pair in smt_module.attributes[KEY_NEED_VERIFY]:
            concrete_funcname, transfer_funcname = func_pair
            transfer_func = func_name_to_func[transfer_funcname.data]
            concrete_func = func_name_to_func[concrete_funcname.data]

            # basic constraint check
            added_ops = basic_constraint_check(transfer_func, get_constraint)
            smt_module.body.block.add_ops(added_ops)
            print("Basic Constraint Check result:", verify_pattern(ctx, smt_module))
            for op in added_ops:
                op.detach()

            # soundness check
            added_ops = soundness_check(
                transfer_func, concrete_func, get_constraint, get_instance_constraint
            )
            smt_module.body.block.add_ops(added_ops)
            print("Soundness Check result:", verify_pattern(ctx, smt_module))
            for op in added_ops:
                op.detach()
        print("")


if __name__ == "__main__":
    main()
