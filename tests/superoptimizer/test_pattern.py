from xdsl.dialects.func import FuncOp
from xdsl.parser import Parser
from xdsl.context import Context

from xdsl.dialects import get_all_dialects
from xdsl.dialects.func import FuncOp

from xdsl_smt.dialects.smt_dialect import SMTDialect
from xdsl_smt.dialects.smt_bitvector_dialect import SMTBitVectorDialect
from xdsl_smt.passes.smt_expand import SMTExpand
from xdsl_smt.passes.lower_to_smt.lower_to_smt import LowerToSMTPass
from xdsl_smt.passes.lower_effects import LowerEffectPass
from xdsl_smt.passes.lower_to_smt.smt_lowerer_loaders import (
    load_vanilla_semantics_using_control_flow_dialects,
)

from xdsl_smt.superoptimization.pattern import Pattern


def create_pattern(source: str, lower_to_smt: bool = True) -> Pattern:
    """Parse a pattern from a string."""
    ctx = Context()
    for dialect_name, dialect_factory in get_all_dialects().items():
        if dialect_name != "smt":
            ctx.register_dialect(dialect_name, dialect_factory)
    ctx.load_dialect(SMTDialect)
    ctx.load_dialect(SMTBitVectorDialect)
    module = Parser(ctx, source).parse_module(True)
    func = module.body.ops.first
    assert isinstance(func, FuncOp)

    semantics_module = module.clone()
    if lower_to_smt:
        load_vanilla_semantics_using_control_flow_dialects()
        LowerToSMTPass().apply(ctx, semantics_module)
        LowerEffectPass().apply(ctx, semantics_module)
        SMTExpand().apply(ctx, semantics_module)

    semantics_func = semantics_module.body.ops.first
    assert isinstance(semantics_func, FuncOp)

    return Pattern(func, semantics_func)


def smt_pattern() -> Pattern:
    """An example SMT pattern that is just an addition of two integers."""
    source = """
             func.func @main(%a: !smt.bv<32>, %b: !smt.bv<32>) -> !smt.bv<32> {
                 %c = "smt.bv.add"(%a, %b) : (!smt.bv<32>, !smt.bv<32>) -> !smt.bv<32>
                 return %c : !smt.bv<32>
             }
             """
    return create_pattern(source, False)


def add_pattern() -> Pattern:
    """An example pattern that is just an addition of two integers."""
    source = """
             func.func @main(%a: i32, %b: i32) -> i32 {
                 %c = arith.addi %a, %b : i32
                 return %c : i32
             }
             """
    return create_pattern(source)


def two_add_pattern() -> Pattern:
    """An example pattern that is just an addition of two integers."""
    source = """
             func.func @main(%a: i32, %b: i32, %c: i32) -> i32 {
                 %d = arith.addi %a, %b : i32
                 %e = arith.addi %d, %c : i32
                 return %e : i32
             }
             """
    return create_pattern(source)


def useless_parameter_pattern() -> Pattern:
    """An example pattern that is just an addition of two integers."""
    source = """
             func.func @main(%a: i32, %b: i32) -> i32 {
                 return %a : i32
             }
             """
    return create_pattern(source)


def test_size():
    pattern = smt_pattern()
    assert pattern.size == 1

    pattern = add_pattern()
    assert pattern.size == 1

    pattern = two_add_pattern()
    assert pattern.size == 2


def test_useless_parameter():
    pattern = useless_parameter_pattern()
    assert 1 in pattern.useless_parameters


def test_permutations():
    pattern = smt_pattern()
    perms = list(pattern.input_permutations())
    assert len(perms) == 2
    assert (0, 1) in perms
    assert (1, 0) in perms

    pattern = add_pattern()
    perms = list(pattern.input_permutations())
    assert len(perms) == 2
    assert (0, 1, 2) in perms
    assert (1, 0, 2) in perms

    pattern = useless_parameter_pattern()
    perms = list(pattern.input_permutations())
    assert perms == [(0, 1, 2)]
