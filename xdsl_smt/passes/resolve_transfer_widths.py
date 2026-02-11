from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects.builtin import (
    ArrayAttr,
    DictionaryAttr,
    IntegerAttr,
    SymbolRefAttr,
    ModuleOp,
    IndexType,
)
from xdsl.ir import Attribute, Operation, ParametrizedAttribute
from typing import cast
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    PatternRewriteWalker,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.parse_pipeline import PipelinePassSpec

from xdsl_smt.dialects.transfer import TransIntegerType


def _get_width_from_attr(attr: Attribute) -> int:
    if TransIntegerType.is_index_integer_attr(attr):
        width_attr = cast(IntegerAttr[IndexType], attr)
        return width_attr.value.data
    raise VerifyException("Width mapping values must be integer attributes")


def _build_width_map(module: ModuleOp) -> dict[str, int]:
    width_map: dict[str, int] = {}
    if "transfer.widths" not in module.attributes:
        return width_map
    widths_attr = module.attributes["transfer.widths"]
    if not isinstance(widths_attr, DictionaryAttr):
        raise VerifyException("transfer.widths must be a dictionary attribute")
    for key, value in widths_attr.data.items():
        width = _get_width_from_attr(value)
        if width <= 0:
            raise VerifyException("transfer.widths values must be positive")
        width_map[key] = width
    return width_map


def resolve_transfer_widths(
    attr: Attribute, width_map: dict[str, int], default_width: int | None
) -> Attribute:
    if isinstance(attr, TransIntegerType):
        width_attr = attr.width
        if isinstance(width_attr, SymbolRefAttr):
            if len(width_attr.nested_references.data) != 0:
                raise VerifyException(
                    "transfer.integer width symbol must be a root symbol ref"
                )
            sym_name = width_attr.root_reference.data
            if sym_name in width_map:
                return TransIntegerType(IntegerAttr(width_map[sym_name], IndexType()))
            if default_width is None:
                raise VerifyException(
                    f"Unresolved transfer.integer width symbol @{sym_name}"
                )
            return TransIntegerType(IntegerAttr(default_width, IndexType()))
        if isinstance(width_attr, IntegerAttr):
            return attr
        raise VerifyException("transfer.integer has invalid width parameter")

    if isinstance(attr, ArrayAttr):
        attr_typed = cast(ArrayAttr[Attribute], attr)
        data = attr_typed.data
        new_data: list[Attribute] = [
            resolve_transfer_widths(a, width_map, default_width) for a in data
        ]
        if all(a1 is a2 for a1, a2 in zip(new_data, data, strict=True)):
            return attr_typed
        return ArrayAttr(new_data)

    if isinstance(attr, DictionaryAttr):
        new_map: dict[str, Attribute] = {}
        changed = False
        for k, v in attr.data.items():
            new_v = resolve_transfer_widths(v, width_map, default_width)
            if new_v is not v:
                changed = True
            new_map[k] = new_v
        if not changed:
            return attr
        return DictionaryAttr(new_map)

    if isinstance(attr, ParametrizedAttribute):
        new_params = [
            resolve_transfer_widths(p, width_map, default_width)
            for p in attr.parameters
        ]
        if all(p1 is p2 for p1, p2 in zip(new_params, attr.parameters, strict=True)):
            return attr
        return attr.new(new_params)

    return attr


def _parse_width_map_spec(spec: str) -> tuple[dict[str, int], int | None]:
    width_map: dict[str, int] = {}
    default_width: int | None = None
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if entry.startswith("default="):
            value = entry.removeprefix("default=").strip()
            if not value.isdigit():
                raise VerifyException(f"Invalid width in width map: {entry}")
            default_width = int(value)
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
    if default_width is not None and default_width <= 0:
        raise VerifyException("Default width must be positive")
    return width_map, default_width


class ResolveTransferWidthsPattern(RewritePattern):
    def __init__(self, width_map: dict[str, int], default_width: int | None):
        self.width_map = width_map
        self.default_width = default_width

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        for result in tuple(op.results):
            new_type = resolve_transfer_widths(
                result.type, self.width_map, self.default_width
            )
            if new_type != result.type:
                rewriter.replace_value_with_new_type(result, new_type)

        for region in op.regions:
            for block in region.blocks:
                for arg in tuple(block.args):
                    new_type = resolve_transfer_widths(
                        arg.type, self.width_map, self.default_width
                    )
                    if new_type != arg.type:
                        rewriter.replace_value_with_new_type(arg, new_type)

        has_done_action = False
        for name, attr in op.attributes.items():
            new_attr = resolve_transfer_widths(attr, self.width_map, self.default_width)
            if new_attr != attr:
                op.attributes[name] = new_attr
                has_done_action = True
        for name, attr in op.properties.items():
            new_attr = resolve_transfer_widths(attr, self.width_map, self.default_width)
            if new_attr != attr:
                op.properties[name] = new_attr
                has_done_action = True
        if has_done_action:
            rewriter.handle_operation_modification(op)


@dataclass(frozen=True)
class ResolveTransferWidths(ModulePass):
    name = "resolve-transfer-widths"
    width_map: dict[str, int] = field(default_factory=dict)
    default_width: int | None = None

    @classmethod
    def from_pass_spec(cls, spec: PipelinePassSpec) -> "ResolveTransferWidths":
        args = spec.normalize_arg_names().args
        width_map_spec: str | None = None
        if "width_map" in args:
            arg = args.pop("width_map")
            if len(arg) == 0:
                width_map_spec = ""
            else:
                width_map_spec = ",".join(str(v) for v in arg)
        if len(args) != 0:
            arguments_str = ", ".join(f'"{arg}"' for arg in args)
            raise ValueError(
                f"Provided arguments [{arguments_str}] not found in expected pass arguments "
                f"['width_map']"
            )
        if width_map_spec is None:
            return cls()
        width_map, default_width = _parse_width_map_spec(width_map_spec)
        return cls(width_map=width_map, default_width=default_width)

    def apply(self, ctx: Context, op: ModuleOp):
        width_map = _build_width_map(op)
        width_map = {**width_map, **self.width_map}
        default_width = self.default_width
        walker = PatternRewriteWalker(
            ResolveTransferWidthsPattern(width_map, default_width), walk_reverse=True
        )
        walker.rewrite_module(op)
        op.attributes.pop("transfer.widths", None)
