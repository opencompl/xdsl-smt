#!/usr/bin/env python3

import argparse
import sys
from typing import cast

from xdsl.ir import Attribute, MLContext, Operation, SSAValue
from xdsl.parser import XDSLParser

from dialects.smt_bitvector_dialect import BitVectorType, SMTBitVectorDialect
from dialects.smt_dialect import (AndOp, CallOp, DefineFunOp, EqOp, ImpliesOp,
                                  AssertOp, NotOp, SMTDialect, BoolType)
from dialects.smt_bitvector_dialect import SMTBitVectorDialect
from dialects.arith_dialect import Arith
from dialects.smt_utils_dialect import (AnyPairType, FirstOp, PairType,
                                        SMTUtilsDialect, SecondOp)
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.dialects.func import Func

from passes.lower_pairs import LowerPairs
from passes.arith_to_smt import ArithToSMT
from passes.canonicalize_smt import CanonicalizeSMT

from traits.smt_printer import print_to_smtlib


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument("before_file",
                            type=str,
                            nargs="?",
                            help="path to before input file")

    arg_parser.add_argument("after_file",
                            type=str,
                            nargs="?",
                            help="path to after input file")

    arg_parser.add_argument("-opt",
                            help="Optimize the SMTLib program by lowering "
                            "pairs and applying constant folding.",
                            action="store_true")


def parse_file(file: str | None) -> Operation:
    if file is None:
        f = sys.stdin
    else:
        f = open(file)

    parser = XDSLParser(ctx, f.read())
    module = parser.parse_op()
    return module


def is_integer_type(type: Attribute) -> bool:
    """Check if an attribute is the SMT representation of an integer type."""
    if not isinstance(type, PairType):
        return False
    type = cast(AnyPairType, type)
    return isinstance(type.first, BitVectorType) and isinstance(
        type.second, BoolType)


def integer_refinement(
        value: SSAValue,
        value_after: SSAValue) -> tuple[list[Operation], SSAValue]:
    """
    Create operations to check that one integer SSAValue refines another.
    The SSAValue returned is a boolean SSAValue true if there is a refinemnet.
    """
    ops = list[Operation]()

    # Get the integer values, and the UB
    val = FirstOp.from_value(value)
    val_after = FirstOp.from_value(value_after)
    poison = SecondOp.from_value(value)
    poison_after = SecondOp.from_value(value_after)
    ops += [val, val_after, poison, poison_after]

    # The refinement rule is
    # not poison => not poison_after and value == value_after
    not_poison = NotOp.get(poison.res)
    not_poison_after = NotOp.get(poison_after.res)
    eq_vals = EqOp.get(val.res, val_after.res)
    and_ = AndOp.get(eq_vals.res, not_poison_after.res)
    refinement = ImpliesOp.get(not_poison.res, and_.res)
    ops += [not_poison, not_poison_after, eq_vals, and_, refinement]

    return ops, refinement.res


def function_refinement(func: DefineFunOp,
                        func_after: DefineFunOp) -> list[Operation]:
    """
    Create operations to check that one function refines another.
    An assert check is added to the end of the list of operations.
    """
    if (len(func.body.blocks[0].args) != 0
            or len(func_after.body.blocks[0].args) != 0):
        print("Function with arguments are not yet supported")
        exit(1)

    ops = list[Operation]()

    # Call both operations
    func_call = CallOp.get(func.results[0], [])
    func_call_after = CallOp.get(func_after.results[0], [])
    ops += [func_call, func_call_after]

    # Check that if UB was triggered after optimizations, it was
    # triggered before as well.
    get_ub = FirstOp.from_value(func_call.res)
    get_ub_after = FirstOp.from_value(func_call_after.res)
    refines_ub = ImpliesOp.get(get_ub_after.res, get_ub.res)
    ops += [get_ub, get_ub_after, refines_ub]

    # Get the function returns
    func_ret = SecondOp.from_value(func_call.res)
    func_ret_after = SecondOp.from_value(func_call_after.res)
    ops += [func_ret, func_ret_after]

    ret_val = func_ret.res
    ret_val_after = func_ret_after.res

    refinement_result = refines_ub.res

    # Check the refinement for each result
    while not is_integer_type(ret_val.typ):
        # Peel the first results
        first_ret = FirstOp.from_value(ret_val)
        first_ret_after = FirstOp.from_value(ret_val_after)
        ops += [first_ret, first_ret_after]

        # Check the refinement for the first result
        ref_ops, ref_value = integer_refinement(first_ret.res,
                                                first_ret_after.res)
        ops += ref_ops
        refinement_add_first = AndOp.get(refinement_result, ref_value)
        ops += [refinement_add_first]
        refinement_result = refinement_add_first.res

        # Get the next results
        next_res = SecondOp.from_value(ret_val)
        next_res_after = SecondOp.from_value(ret_val_after)
        ops += [next_res, next_res_after]
        ret_val = next_res.res
        ret_val_after = next_res_after.res

    ops.append(AssertOp.get(refinement_result))

    return ops


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

    # Parse the files
    module = parse_file(args.before_file)
    module_after = parse_file(args.after_file)

    assert (isinstance(module, ModuleOp))
    assert (isinstance(module_after, ModuleOp))

    # Convert both module to SMTLib
    ArithToSMT().apply(ctx, module)
    ArithToSMT().apply(ctx, module_after)

    # Collect the function from both modules
    if (len(module.ops) != len(module_after.ops)
            or not isinstance(module.ops[0], DefineFunOp)
            or not isinstance(module_after.ops[0], DefineFunOp)):
        print("Input is expected to have a single `func.func` operation.")
        exit(1)

    func = module.ops[0]
    func_after = module_after.ops[0]

    # Combine both modules into a new one
    new_module = ModuleOp.from_region_or_ops([])
    block = new_module.body.blocks[0]
    func.detach()
    block.add_op(func)
    func_after.detach()
    block.add_op(func_after)

    # Add refinement operations
    block.add_ops(function_refinement(func, func_after))

    if args.opt:
        LowerPairs().apply(ctx, new_module)
        CanonicalizeSMT().apply(ctx, new_module)
    print_to_smtlib(new_module, sys.stdout)
