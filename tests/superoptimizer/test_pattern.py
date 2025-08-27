from xdsl.dialects.func import FuncOp
from xdsl.parser import Parser
from xdsl.context import Context

from xdsl.dialects import get_all_dialects
from xdsl.dialects.func import FuncOp

from xdsl_smt.passes.smt_expand import SMTExpand
from xdsl_smt.passes.lower_to_smt.lower_to_smt import LowerToSMTPass
from xdsl_smt.passes.lower_effects import LowerEffectPass
from xdsl_smt.passes.lower_to_smt.smt_lowerer_loaders import (
    load_vanilla_semantics_using_control_flow_dialects,
)

from xdsl_smt.superoptimization.pattern import Pattern


def create_pattern(source: str) -> Pattern:
    """Parse a pattern from a string."""
    ctx = Context()
    for dialect_name, dialect_factory in get_all_dialects().items():
        ctx.register_dialect(dialect_name, dialect_factory)
    module = Parser(ctx, source).parse_module(True)
    func = module.body.ops.first
    assert isinstance(func, FuncOp)

    semantics_module = module.clone()
    load_vanilla_semantics_using_control_flow_dialects()
    LowerToSMTPass().apply(ctx, semantics_module)
    LowerEffectPass().apply(ctx, semantics_module)
    SMTExpand().apply(ctx, semantics_module)
    semantics_func = semantics_module.body.ops.first
    assert isinstance(semantics_func, FuncOp)

    return Pattern(func, semantics_func)


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
    pattern = add_pattern()
    assert pattern.size == 1

    pattern = two_add_pattern()
    assert pattern.size == 2


def test_useless_parameter():
    pattern = useless_parameter_pattern()
    assert 1 in pattern.useless_parameters
