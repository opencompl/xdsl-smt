from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import (
    IntegerAttr,
    SymbolRefAttr,
    ModuleOp,
    IndexType,
    NoneAttr,
    FunctionType,
)
from xdsl.dialects.func import FuncOp
from xdsl.ir import Attribute, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    PatternRewriteWalker,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.parse_pipeline import PipelinePassSpec

from xdsl_smt.dialects.transfer import AbstractValueType, TransIntegerType


def _resolve_symbolic_widths(attr: Attribute, width_map: dict[str, int]) -> Attribute:
    if isinstance(attr, TransIntegerType):
        assert not isinstance(attr.width, NoneAttr)
        if isinstance(attr.width, SymbolRefAttr):
            assert len(attr.width.nested_references.data) == 0
            sym_name = attr.width.root_reference.data
            assert sym_name in width_map
            return TransIntegerType(IntegerAttr(width_map[sym_name], IndexType()))
        assert isinstance(attr.width, IntegerAttr)
        return attr
    if isinstance(attr, AbstractValueType):
        new_fields = tuple(
            _resolve_symbolic_widths(t, width_map) for t in attr.fields.data
        )
        if new_fields == attr.fields.data:
            return attr
        return AbstractValueType(list(new_fields))

    return attr


def _resolve_legacy_widths(attr: Attribute, width: int) -> Attribute:
    if isinstance(attr, TransIntegerType):
        assert not isinstance(attr.width, SymbolRefAttr)
        if isinstance(attr.width, NoneAttr):
            return TransIntegerType(IntegerAttr(width, IndexType()))
        assert isinstance(attr.width, IntegerAttr)
        return attr
    if isinstance(attr, AbstractValueType):
        new_fields = tuple(_resolve_legacy_widths(t, width) for t in attr.fields.data)
        if new_fields == attr.fields.data:
            return attr
        return AbstractValueType(list(new_fields))

    return attr


def _parse_width_map_spec(spec: str) -> dict[str, int]:
    width_map: dict[str, int] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if not entry.startswith("@") or "=" not in entry:
            raise VerifyException(f"Invalid map entry '{entry}', expected '@Sym=Width'")
        sym, value = entry.split("=", 1)
        sym = sym.strip()
        value = value.strip()
        if not sym.startswith("@") or sym == "@":
            raise VerifyException(f"Invalid symbol name in width map entry: {entry}")
        if not value.isdigit():
            raise VerifyException(f"Invalid width in width map entry: {entry}")
        width = int(value)
        if width <= 0:
            raise VerifyException("Width mapping values must be positive")
        width_map[sym.removeprefix("@")] = width
    return width_map


class ResolveSymbolicWidthsPattern(RewritePattern):
    def __init__(self, width_map: dict[str, int]):
        self.width_map = width_map

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        for result in tuple(op.results):
            new_type = _resolve_symbolic_widths(result.type, self.width_map)
            if new_type != result.type:
                rewriter.replace_value_with_new_type(result, new_type)

        for region in op.regions:
            for block in region.blocks:
                for arg in tuple(block.args):
                    new_type = _resolve_symbolic_widths(arg.type, self.width_map)
                    if new_type != arg.type:
                        rewriter.replace_value_with_new_type(arg, new_type)

        has_done_action = False
        for name, attr in op.attributes.items():
            new_attr = _resolve_symbolic_widths(attr, self.width_map)
            if new_attr != attr:
                op.attributes[name] = new_attr
                has_done_action = True
        for name, attr in op.properties.items():
            new_attr = _resolve_symbolic_widths(attr, self.width_map)
            if new_attr != attr:
                op.properties[name] = new_attr
                has_done_action = True
        if isinstance(op, FuncOp):
            new_inputs = tuple(arg.type for arg in op.body.block.args)
            new_outputs = tuple(
                _resolve_symbolic_widths(t, self.width_map)
                for t in op.function_type.outputs
            )
            new_type = FunctionType.from_lists(new_inputs, new_outputs)
            if new_type != op.function_type:
                op.function_type = new_type
                has_done_action = True
        if has_done_action:
            rewriter.handle_operation_modification(op)


class ResolveLegacyWidthsPattern(RewritePattern):
    def __init__(self, width: int):
        self.width = width

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        for result in tuple(op.results):
            new_type = _resolve_legacy_widths(result.type, self.width)
            if new_type != result.type:
                rewriter.replace_value_with_new_type(result, new_type)

        for region in op.regions:
            for block in region.blocks:
                for arg in tuple(block.args):
                    new_type = _resolve_legacy_widths(arg.type, self.width)
                    if new_type != arg.type:
                        rewriter.replace_value_with_new_type(arg, new_type)

        has_done_action = False
        for name, attr in op.attributes.items():
            new_attr = _resolve_legacy_widths(attr, self.width)
            if new_attr != attr:
                op.attributes[name] = new_attr
                has_done_action = True
        for name, attr in op.properties.items():
            new_attr = _resolve_legacy_widths(attr, self.width)
            if new_attr != attr:
                op.properties[name] = new_attr
                has_done_action = True
        if isinstance(op, FuncOp):
            new_inputs = tuple(arg.type for arg in op.body.block.args)
            new_outputs = tuple(
                _resolve_legacy_widths(t, self.width) for t in op.function_type.outputs
            )
            new_type = FunctionType.from_lists(new_inputs, new_outputs)
            if new_type != op.function_type:
                op.function_type = new_type
                has_done_action = True
        if has_done_action:
            rewriter.handle_operation_modification(op)


@dataclass(frozen=True)
class ResolveTransferWidths(ModulePass):
    name = "resolve-transfer-widths"
    width_map: dict[str, int] = field(default_factory=dict)
    width: int | None = None

    @classmethod
    def from_pass_spec(cls, spec: PipelinePassSpec) -> "ResolveTransferWidths":
        args = spec.normalize_arg_names().args
        if len(args) == 1:
            ((only_key, only_val),) = args.items()
            if only_key.isdigit() and len(only_val) == 0:
                return cls(width=int(only_key))
        width_map_spec: str | None = None
        width_value: int | None = None
        if "width_map" in args:
            arg = args.pop("width_map")
            if len(arg) == 0:
                width_map_spec = ""
            else:
                width_map_spec = ",".join(str(v) for v in arg)
        if "width" in args:
            arg = args.pop("width")
            if len(arg) != 1:
                raise ValueError('Expected a single value for "width"')
            value = str(arg[0]).strip()
            if not value.isdigit():
                raise ValueError(f'Invalid width value "{value}"')
            width_value = int(value)
            if width_value <= 0:
                raise ValueError("Width must be positive")
        if width_map_spec is not None and width_value is not None:
            raise ValueError('Args "width_map" and "width" are exclusive')
        if len(args) != 0:
            args_str = ", ".join(f'"{arg}"' for arg in args)
            raise ValueError(
                f"Args [{args_str}] not found in expected ['width_map', 'width']"
            )
        if width_map_spec is None:
            return cls(width=width_value)
        width_map = _parse_width_map_spec(width_map_spec)
        return cls(width_map=width_map, width=width_value)

    def apply(self, ctx: Context, op: ModuleOp):
        if self.width is not None:
            if self.width <= 0:
                raise VerifyException("Width must be positive")
            walker = PatternRewriteWalker(
                ResolveLegacyWidthsPattern(self.width), walk_reverse=True
            )
            walker.rewrite_module(op)
            return

        walker = PatternRewriteWalker(
            ResolveSymbolicWidthsPattern(self.width_map), walk_reverse=True
        )
        walker.rewrite_module(op)
