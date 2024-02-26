#!/usr/bin/env python3

import argparse

from xdsl.ir import MLContext, Operation
from xdsl.parser import Parser

from ..dialects.smt_dialect import SMTDialect
from ..dialects.smt_bitvector_dialect import SMTBitVectorDialect
from ..dialects.index_dialect import Index
from ..dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func, FuncOp
from xdsl.dialects.arith import Arith
from ..dialects.transfer import Transfer, AbstractValueType

from ..passes import calculate_smt as cs
from ..utils.trans_interpreter_smt import *
from ..passes.rename_values import RenameValuesPass

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


class Oracle:
    @staticmethod
    def get_abs_constraint(func, get_constraint, solver):
        arg_constraint = []
        for i, arg in enumerate(func.body.blocks[0].args):
            if isinstance(arg.type, AbstractValueType):
                smt_val = cs.get_smt_val(arg)
                constraint = get_constraint(solver, smt_val)
                assert constraint is not None
                arg_constraint.append(constraint)
        return arg_constraint

    @staticmethod
    def get_inst_operands(func: FuncOp, get_instance_constraint, solver, width):
        inst_list = []
        inst_constraints = []
        for i, arg in enumerate(func.body.blocks[0].args):
            if isinstance(arg.type, AbstractValueType):
                inst = BitVec(arg.name_hint + "inst", width)
                smt_val = cs.get_smt_val(arg)
                constraint = get_instance_constraint(solver, smt_val, inst)
                assert constraint is not None
                inst_constraints.append(constraint)
                inst_list.append(inst)
            else:
                inst_list.append(getattr(arg, "smtVal"))
        return inst_list, inst_constraints


def basic_constraint_check(absOp, ctx, get_constraint, func_name_to_func, width):
    s = Solver()
    cs.init()
    cs.ToSMTAnalysisPass(func_name_to_func, s, width).apply(ctx, absOp)
    abs_constraint = Oracle.get_abs_constraint(absOp, get_constraint, s)
    s.add(And(abs_constraint))
    key = absOp.sym_name.data + ".return"
    assert key in cs.resultToSMTValue
    smt_val = cs.resultToSMTValue[key]
    constraint = get_constraint(s, smt_val)
    s.add(Not(And(constraint)))
    check_res = s.check()
    if str(check_res) == "sat":
        print("basic constraint check failed!\ncounterexample:\n")
        print(s.model())
        return -1
    elif str(check_res) == "unsat":
        print("basic constraint check successfully")
        return 1
    else:
        print("unknown: ", check_res)
        return 0


def soundness_check(
    concrete_op,
    abs_op,
    ctx,
    get_constraint,
    get_instance_constraint,
    func_name_to_func,
    width,
):
    s = Solver()
    cs.init()
    cs.ToSMTAnalysisPass(func_name_to_func, s, width).apply(ctx, abs_op)
    concrete_func = parse_function_to_python(
        concrete_op, func_name_to_func, cs.opToSMTFunc, cs.funcCall
    )
    abs_constraint = Oracle.get_abs_constraint(abs_op, get_constraint, s)
    s.add(And(abs_constraint))
    key = abs_op.sym_name.data + ".return"
    assert key in cs.resultToSMTValue
    abs_op_result = cs.resultToSMTValue[key]
    assert abs_op_result is not None
    inst_list, inst_constraints = Oracle.get_inst_operands(
        abs_op, get_instance_constraint, s, width
    )
    s.add(inst_constraints)

    inst_result_smt = concrete_func(s, *inst_list)

    constraint = get_instance_constraint(s, abs_op_result, inst_result_smt)

    s.add(simplify(Not(And(constraint))))
    check_res = s.check()
    if str(check_res) == "sat":
        print("soundness check failed!\ncounterexample:\n")
        print(s.model())
        return -1
    elif str(check_res) == "unsat":
        print("soundness check successfully")
        return 1
    else:
        print("unknown: ", check_res)
        return 0


def precision_check(
    concrete_op: FuncOp,
    abs_op,
    ctx,
    get_constraint,
    get_instance_constraint,
    func_name_to_func,
    width,
):
    s = Solver()
    cs.ToSMTAnalysisPass(func_name_to_func, s, width).apply(ctx, abs_op)

    concrete_func = parse_function_to_python(
        concrete_op, func_name_to_func, cs.opToSMTFunc, cs.funcCall
    )
    abs_constraint = Oracle.get_abs_constraint(abs_op, get_constraint, s)
    s.add(And(abs_constraint))
    key = abs_op.sym_name.data + ".return"
    assert key in cs.resultToSMTValue
    abs_result = cs.resultToSMTValue[key]
    assert abs_result is not None
    abs_result_constraint = get_constraint(s, abs_result)
    s.add(abs_result_constraint)

    abs_result_inst = BitVec("absResultInst", width)
    abs_result_inst_constraint = get_instance_constraint(s, abs_result, abs_result_inst)
    s.add(abs_result_inst_constraint)

    inst_list, inst_constraints = Oracle.get_inst_operands(
        abs_op, get_instance_constraint, s, width
    )
    concrete_result = concrete_func(s, *inst_list)
    # for now
    qualifier = inst_list
    s.add(
        ForAll(
            qualifier,
            Implies(And(inst_constraints), abs_result_inst != concrete_result),
        ),
    )
    check_res = s.check()
    if str(check_res) == "sat":
        print("precision check failed!\ncounterexample:\n")
        print(s.model())
        return -1
    elif str(check_res) == "unsat":
        print("precision check successfully")
        return 1
    else:
        print("unknown: ", check_res)
        return 0


KEY_NEED_VERIFY = "builtin.NEED_VERIFY"
MAXIMAL_VERIFIED_BITS = 8


def main() -> None:
    ctx = MLContext()
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    # Register all dialects
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(SMTDialect)
    ctx.load_dialect(SMTBitVectorDialect)
    ctx.load_dialect(SMTUtilsDialect)
    ctx.load_dialect(Transfer)
    ctx.load_dialect(Index)

    # Parse the files
    module = parse_file(ctx, args.transfer_functions)
    assert isinstance(module, ModuleOp)

    get_constraint = None
    get_instance_constraint = None

    func_name_to_func = {}

    RenameValuesPass().apply(ctx, module)
    for func in module.ops:
        if isinstance(func, FuncOp):
            func_name_to_func[func.sym_name.data] = func
            if func.sym_name.data == "getConstraint":
                get_constraint = func
            elif func.sym_name.data == "getInstanceConstraint":
                get_instance_constraint = func

    get_constraint = parse_function_to_python(
        get_constraint, func_name_to_func, cs.opToSMTFunc, cs.funcCall
    )
    get_instance_constraint = parse_function_to_python(
        get_instance_constraint, func_name_to_func, cs.opToSMTFunc, cs.funcCall
    )

    NEED_VERIFY = []

    for pair in module.attributes[KEY_NEED_VERIFY]:
        assert len(pair) == 2 and "concrete and abstract operation should be paired"
        NEED_VERIFY.append([])
        for p in pair:
            NEED_VERIFY[-1].append(p.data)
            assert p.data in func_name_to_func and "Cannot find the specified function"
        for width in range(1, MAXIMAL_VERIFIED_BITS + 1):
            print(
                "Currently verifying: ", NEED_VERIFY[-1][0], " with bitwidth, ", width
            )
            basic_constraint_check(
                func_name_to_func[NEED_VERIFY[-1][1]],
                ctx,
                get_constraint,
                func_name_to_func,
                width,
            )
            soundness_check(
                func_name_to_func[NEED_VERIFY[-1][0]],
                func_name_to_func[NEED_VERIFY[-1][1]],
                ctx,
                get_constraint,
                get_instance_constraint,
                func_name_to_func,
                width,
            )
            precision_check(
                func_name_to_func[NEED_VERIFY[-1][0]],
                func_name_to_func[NEED_VERIFY[-1][1]],
                ctx,
                get_constraint,
                get_instance_constraint,
                func_name_to_func,
                width,
            )


if __name__ == "__main__":
    main()
