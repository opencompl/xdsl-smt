from abc import ABCMeta
from typing import (Any, Callable, Iterable, Self, TypeVar)

from xdsl.ir import (Operation, OpResult)
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)

_debug = False

if _debug:
    from sys import stderr

    def _debugp(s: str) -> None:
        print(s, file=stderr)
else:
    def _debugp(s: str) -> None:
        pass


def new_ops(op: Operation) -> Iterable[Operation]:
    """Iterate over an un-parented operation and its operands in post-order (operands first)"""

    if op.parent is None:
        for child_op in op.operands:
            if isinstance(child_op, OpResult):
                yield from new_ops(child_op.op)
        yield op


class PatternRegistrar:
    """Maintains a registry (list) of rewrite patterns

    Instances can be used as a class decorator to automatically create an instance and register it
    as a rewrite pattern. The instance's zero-argument initializer is used."""

    _RewritePatternT = TypeVar("_RewritePatternT", bound=RewritePattern)

    registry: list[RewritePattern]

    def __init__(self: Self) -> None:
        self.registry = []

    def register_rewrite_pattern(self: Self, pattern: RewritePattern) -> None:
        """Register a rewrite pattern"""

        self.registry.append(pattern)

    def __call__(self: Self, cls: type[_RewritePatternT]) -> type[_RewritePatternT]:
        """Automatically register a rewrite pattern of the provided type"""

        self.register_rewrite_pattern(cls())

        return cls


class SimpleRewriteMeta(ABCMeta):
    """Metaclass for more concisely defining rewrites

    Classes with this metaclass are expected to define a `source` attribute
    specifying the source operation and a `rewrite` attribute that takes an
    instance of the source operation and returns the rewritten operation."""

    def __new__(cls: type[Self], name: str, bases: tuple[type, ...], attrs: dict[str, Any]):
        try:
            srcOp: type[Operation] = attrs['source']
            rewrite: Callable[[Operation], Operation] = attrs['rewrite']
        except KeyError as e:
            _debugp(f"{name} has no {e}")

            return super().__new__(cls, name, bases, attrs)

        @op_type_rewrite_pattern
        def match_and_rewrite(self: RewritePattern, op: srcOp, rewriter: PatternRewriter) -> None: # type: ignore
            rewriter.replace_matched_op([*new_ops(rewrite(op))]) # type: ignore

        return super().__new__(cls, name, bases, {'match_and_rewrite': match_and_rewrite, **attrs})


class SimpleRewritePattern(RewritePattern, metaclass=SimpleRewriteMeta):
    """Common base class for rewrites"""

    pass


class SimpleBinOpRewriteMeta(SimpleRewriteMeta):
    """Metaclass for more concisely defining binop→binop rewrites

    Classes with this metaclass are expected to define a `source` attribute
    specifying the source operation and a `target` attribute specifying the
    target operation."""

    def __new__(cls: type[Self], name: str, bases: tuple[type, ...], attrs: dict[str, Any]):
        try:
            srcOp: type[Operation] = attrs['source']
            tgtOp: type[Operation] = attrs['target']
        except KeyError:
            return super().__new__(cls, name, bases, attrs)

        def rewrite(op: srcOp) -> tgtOp: # type: ignore
            return tgtOp(op.lhs, op.rhs) # type: ignore

        return super().__new__(cls, name, bases, {'rewrite': rewrite, **attrs})


class SimpleBinOpRewritePattern(SimpleRewritePattern, metaclass=SimpleBinOpRewriteMeta):
    """Common base class for binop rewrites"""

    pass


class SimpleRewritePatternFactory:
    """Convenient way to create simple rewrites and automatically add their classes to the module's
    scope"""

    registrar: PatternRegistrar
    scope: dict[str, Any]

    def __init__(self: Self, registrar: PatternRegistrar, scope: dict[str, Any]) -> None:
        self.registrar = registrar
        self.scope = scope

    def _make(self: Self,
              name: str,
              base: type[RewritePattern],
              attrs: dict[str, Any]) -> type[RewritePattern]:
        cls = self.registrar(type(name, (base, ), attrs))
        self.scope[name] = cls
        return cls


    def make_simple(self: Self,
                    srcOp: type[Operation],
                    rewrite: Callable[[Operation], Operation]) -> None:
        """Make a rewrite pattern from `srcOp` to `rewrite(srcOp)`"""

        self._make(srcOp.__name__ + 'RewritePattern',
                   SimpleRewritePattern,
                   {'source': srcOp, 'rewrite': rewrite})


    def make_binop(self: Self, srcOp: type[Operation], tgtOp: type[Operation]) -> None:
        """Make a rewrite pattern from `srcOp` to `tgtOp`"""

        self._make(srcOp.__name__ + 'RewritePattern',
                   SimpleBinOpRewritePattern,
                   {'source': srcOp, 'target': tgtOp})
