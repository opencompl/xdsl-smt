"""
Add all implicitely matched and rewritten properties in PDL patterns.
* When an operation is matched, all its properties that are not specified
  are added to the pattern so they are matched explicitely.
* When an operation is inserted by the pattern, all its properties that are not
  specified and that have a default value are added to the pattern. Having
  non-specified properties without a default value will result in an error.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.irdl import IRDLOperation, PropertyDef
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
    PatternRewriteWalker,
)

from xdsl.dialects import pdl
from xdsl.dialects.pdl import OperationOp
from xdsl.dialects.builtin import ModuleOp, StringAttr, ArrayAttr


def add_implicitely_matched_attribute(
    op: pdl.OperationOp,
    prop_name: str,
    prop_def: PropertyDef,
    rewriter: PatternRewriter,
) -> pdl.OperationOp:
    """
    Add an implicitely matched attribute to a matched operation.
    This will create a new `pdl.attribute` with a constraint that the attribute
    is satisfying the property definition.
    """
    # For now, only handle properties that are of a given attribute type.
    base = prop_def.constr.get_unique_base()
    if base is None:
        raise ValueError(
            f"Property {prop_name} of operation {op.opName} does not have a unique "
            "base type."
        )

    # Create a new attribute matcher
    attr_op = rewriter.insert(pdl.AttributeOp(None))
    attr_op.attributes["baseType"] = StringAttr(base.name)

    # Update the operation to add the attribute
    new_op = pdl.OperationOp(
        op.opName,
        ArrayAttr((*op.attributeValueNames.data, StringAttr(prop_name))),
        op.operand_values,
        (*op.attribute_values, attr_op.output),
        op.type_values,
    )
    rewriter.replace_op(op, new_op)
    return new_op


def add_implicitely_inserted_attribute(
    op: pdl.OperationOp,
    prop_name: str,
    prop_def: PropertyDef,
    rewriter: PatternRewriter,
) -> pdl.OperationOp:
    """
    Add an implicitely inserted attribute to an inserted operation.
    This will create a new `pdl.attribute` with the default property value
    defined in the operation definition.
    """
    # For now, only handle properties that are of a given attribute type.
    default = prop_def.default_value
    if default is None:
        raise ValueError(
            f"Property {prop_name} of operation {op.opName} does not have a "
            "default value"
        )

    # Create a new attribute matcher
    attr_op = rewriter.insert(pdl.AttributeOp(default))

    # Update the operation to add the attribute
    new_op = pdl.OperationOp(
        op.opName,
        ArrayAttr((*op.attributeValueNames.data, StringAttr(prop_name))),
        op.operand_values,
        (*op.attribute_values, attr_op.output),
        op.type_values,
    )
    rewriter.replace_op(op, new_op)
    return new_op


@dataclass
class AddImplicitPropertiesPattern(RewritePattern):
    ctx: Context

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: OperationOp, rewriter: PatternRewriter):
        # Check if the operation is an IRDLOperation that is registered,
        # and grab its IRDL definition.
        op_name = op.opName
        if op_name is None:
            return
        if (op_type := self.ctx.get_optional_op(op_name.data)) is None:
            return
        if not issubclass(op_type, IRDLOperation):
            return
        op_def = op_type.get_irdl_definition()

        # Check if the operation has properties that are not specified.
        for prop_name, prop_def in op_def.properties.items():
            if prop_name not in [attr.data for attr in op.attributeValueNames.data]:
                if isinstance(op.parent_op(), pdl.PatternOp):
                    add_implicitely_matched_attribute(op, prop_name, prop_def, rewriter)
                else:
                    assert isinstance(op.parent_op(), pdl.RewriteOp)
                # Return as the operation is now invalidated.
                # Next applications of the pattern will fix the other properties.
                return


class PDLAddImplicitPropertiesPass(ModulePass):
    name = "pdl-add-implicit-properties"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(AddImplicitPropertiesPattern(ctx)).rewrite_module(op)
