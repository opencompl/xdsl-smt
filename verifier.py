#!/usr/bin/env python3

import argparse
import sys

from xdsl.ir import MLContext, Operation
from xdsl.parser import MLIRParser

from dialects.smt_bitvector_dialect import SMTBitVectorDialect
from dialects.smt_dialect import CallOp, DefineFunOp, EqOp, AssertOp, SMTDialect
from dialects.smt_bitvector_dialect import SMTBitVectorDialect
from dialects.arith_dialect import Arith
from dialects.index_dialect import Index
from dialects.smt_utils_dialect import SMTUtilsDialect
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func, FuncOp
from xdsl.dialects.transfer import Transfer, AbstractValueType

from passes.lower_pairs import LowerPairs
from passes.arith_to_smt import ArithToSMT
from passes.calculate_smt import CalculateSMT,WIDTH
import passes.calculate_smt
from passes.canonicalize_smt import CanonicalizeSMT

from traits.smt_printer import print_to_smtlib
from z3 import *


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "transfer_functions", type=str, nargs="?", help="path to the transfer functions"
    )

    arg_parser.add_argument(
        "-opt",
        help="Optimize the SMTLib program by lowering "
             "pairs and applying constant folding.",
        action="store_true",
    )


def parse_file(file: str | None) -> Operation:
    if file is None:
        f = sys.stdin
    else:
        f = open(file)

    parser = MLIRParser(ctx, f.read())
    module = parser.parse_op()
    return module


def callSMTFunction(func: FuncOp, args, ctx):
    setattr(func, "smtArgs", args)
    CalculateSMT().apply(ctx, func)
    return getattr(func.get_return_op(), "smtVal")


class Oracle:
    @staticmethod
    def getAbsConstraint(func, ctx, getConstraint: FuncOp):
        argConstraint = []
        for i, arg in enumerate(func.body.blocks[0].args):
            if isinstance(arg.typ, AbstractValueType):
                constraint = callSMTFunction(getConstraint, [getattr(arg, "smtVal")], ctx)
                assert constraint is not None
                argConstraint.append(constraint)
        return argConstraint

    @staticmethod
    def getInstOperands(func: FuncOp, ctx, getInstanceConstraint: FuncOp, width):
        instList = []
        instConstraints = []
        for i, arg in enumerate(func.body.blocks[0].args):
            if isinstance(arg.typ, AbstractValueType):
                inst = BitVec(arg.name + "inst", width)
                constraint = callSMTFunction(getInstanceConstraint, [getattr(arg, "smtVal"), inst], ctx)
                assert constraint is not None
                instConstraints.append(constraint)
                instList.append(inst)
            else:
                instList.append(getattr(arg, "smtVal"))
        return (instList, instConstraints)


def basicConstraintCheck(absOp: FuncOp, ctx, getConstraint: FuncOp):
    s = Solver()
    absConstraint = Oracle.getAbsConstraint(absOp, ctx, getConstraint)
    s.add(And(absConstraint))
    smtVal = getattr(absOp.get_return_op(), "smtVal")
    assert smtVal is not None
    constraint = callSMTFunction(getConstraint, [smtVal], ctx)
    s.add(Not(And(constraint)))
    checkRes = s.check()
    if str(checkRes) == 'sat':
        print("basic constraint check failed!\ncounterexample:\n")
        print(s.model())
        return -1
    elif str(checkRes) == 'unsat':
        print("basic constraint check successfully")
        return 1
    else:
        print("unknown: ", checkRes)
        return 0


def soundnessCheck(concreteOp: FuncOp, absOp: FuncOp, ctx, getConstraint: FuncOp, getInstanceConstraint: FuncOp, width):
    s = Solver()
    absConstraint = Oracle.getAbsConstraint(absOp, ctx, getConstraint)
    s.add(And(absConstraint))
    absOpResult = getattr(absOp.get_return_op(), "smtVal")
    assert absOpResult is not None
    instList, instConstraints = Oracle.getInstOperands(absOp, ctx, getInstanceConstraint, width)
    s.add(instConstraints)

    instResultSMT = callSMTFunction(concreteOp, instList, ctx)

    constraint = callSMTFunction(getInstanceConstraint, [absOpResult, instResultSMT], ctx)

    s.add(simplify(Not(And(constraint))))
    checkRes = s.check()
    if str(checkRes) == 'sat':
        print("soundness check failed!\ncounterexample:\n")
        print(s.model())
        return -1
    elif str(checkRes) == 'unsat':
        print("soundness check successfully")
        return 1
    else:
        print("unknown: ", checkRes)
        return 0


def precisionCheck(concreteOp: FuncOp, absOp: FuncOp, ctx, getConstraint: FuncOp, getInstanceConstraint: FuncOp, width):
    s = Solver()
    absContraint = Oracle.getAbsConstraint(absOp, ctx, getConstraint)
    s.add(absContraint)

    absResult = getattr(absOp.get_return_op(), "smtVal")
    absResultConstraint = callSMTFunction(getConstraint, [absResult], ctx)
    s.add(absResultConstraint)

    absResultInst = BitVec("absResultInst", width)
    absResultInstConstraint = callSMTFunction(getInstanceConstraint, [absResult, absResultInst], ctx)
    s.add(absResultInstConstraint)

    instList, instConstraints = Oracle.getInstOperands(absOp, ctx, getInstanceConstraint, width)
    concreteResult = callSMTFunction(concreteOp, instList, ctx)
    # for now
    qualifier = instList
    s.add(
        ForAll(qualifier,
               Implies(And(instConstraints),
                       absResultInst != concreteResult))
    )
    checkRes = s.check()
    if str(checkRes) == 'sat':
        print("precision check failed!\ncounterexample:\n")
        print(s.model())
        return -1
    elif str(checkRes) == 'unsat':
        print("precision check successfully")
        return 1
    else:
        print("unknown: ", checkRes)
        return 0


KEY_NEED_VERIFY = "builtin.NEED_VERIFY"
MAXIMAL_VERIFIED_BITS=32

if __name__ == "__main__":
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
    module = parse_file(args.transfer_functions)
    assert isinstance(module, ModuleOp)

    getConstraint = None
    getInstanceConstraint = None

    funcNameToFunc = {}
    NEED_VERIFY = ("OR", "ORImpl")
    for func in module.ops:
        if isinstance(func, FuncOp):
            funcNameToFunc[func.sym_name.data] = func
            if func.sym_name.data == "getConstraint":
                getConstraint = func
            elif func.sym_name.data == "getInstanceConstraint":
                getInstanceConstraint = func

    NEED_VERIFY = []

    for pair in module.attributes[KEY_NEED_VERIFY]:
        assert len(pair) == 2 and "concrete and abstract operation should be paired"
        NEED_VERIFY.append([])
        for p in pair:
            NEED_VERIFY[-1].append(p.data)
            assert p.data in funcNameToFunc and "Cannot find the specified function"
        print("Currently verifying: ", NEED_VERIFY[-1][0])
        for i in range(1,MAXIMAL_VERIFIED_BITS+1):
            passes.calculate_smt.WIDTH = i
            CalculateSMT().apply(ctx, funcNameToFunc[NEED_VERIFY[-1][0]])
            CalculateSMT().apply(ctx, funcNameToFunc[NEED_VERIFY[-1][1]])
            basicConstraintCheck(funcNameToFunc[NEED_VERIFY[-1][1]], ctx, getConstraint)
            soundnessCheck(funcNameToFunc[NEED_VERIFY[-1][0]], funcNameToFunc[NEED_VERIFY[-1][1]], ctx, getConstraint,
                           getInstanceConstraint, i)
            precisionCheck(funcNameToFunc[NEED_VERIFY[-1][0]], funcNameToFunc[NEED_VERIFY[-1][1]], ctx, getConstraint,
                           getInstanceConstraint, i)
