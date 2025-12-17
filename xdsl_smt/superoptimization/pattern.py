"""
This file defines the Pattern class, which represent a DAG with holes using a
function. The Pattern class additionally stores the semantics of the DAG, as
well as additional semantic information.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast, Iterable
import itertools
import z3  # pyright: ignore[reportMissingTypeStubs]

from xdsl.ir import SSAValue, BlockArgument, OpResult, Attribute
from xdsl.context import Context
from xdsl.builder import Builder
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.utils.hints import isa

from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.dialects.builtin import IntegerAttr, ArrayAttr, IntAttr, ModuleOp

from xdsl_smt.passes.transfer_inline import FunctionCallInline
from xdsl_smt.dialects import (
    smt_dialect as smt,
    smt_bitvector_dialect as bv,
    smt_utils_dialect as pair,
)
from xdsl_smt.cli.xdsl_smt_run import build_interpreter, interpret
from xdsl_smt.utils.run_with_smt_solver import run_module_through_smtlib
from xdsl_smt.utils.frozen_multiset import FrozenMultiset
from xdsl_smt.utils.permutation import Permutation, permute
from xdsl_smt.semantics.refinements import function_results_refinement


def values_of_type(ty: Attribute) -> tuple[list[Attribute], bool]:
    """
    Returns values of the passed type.

    The boolean indicates whether the returned values cover the whole type. If
    `true`, this means all possible values of the type were returned.
    """
    match ty:
        case smt.BoolType():
            return [IntegerAttr.from_bool(False), IntegerAttr.from_bool(True)], True
        case bv.BitVectorType(width=IntAttr(data=width)):
            w = min(2, width)
            return [
                IntegerAttr.from_int_and_width(value, width) for value in range(1 << w)
            ], (w == width)
        case pair.PairType():
            ty = cast(pair.PairType, ty)
            first_values, first_total = values_of_type(ty.first)
            second_values, second_total = values_of_type(ty.second)
            return (
                [ArrayAttr([fv, sv]) for fv in first_values for sv in second_values],
                first_total and second_total,
            )
        case _:
            raise ValueError(f"Unsupported type: {ty}")


def clone_func_to_smt_func(func: FuncOp) -> smt.DefineFunOp:
    """
    Convert a `func.func` to an `smt.define_fun` operation.
    Do not mutate the original function.
    """
    new_region = func.body.clone()

    # Replace the `func.return` with an `smt.return` operation.
    func_return = new_region.block.last_op
    assert isinstance(func_return, ReturnOp)
    rewriter = Rewriter()
    rewriter.insert_op(
        smt.ReturnOp(func_return.arguments), InsertPoint.before(func_return)
    )
    rewriter.erase_op(func_return)

    return smt.DefineFunOp(new_region)


Result = tuple[Any, ...]
UnorderedFingerprint = FrozenMultiset[tuple[Result, ...]]


@dataclass(init=False, unsafe_hash=True)
class Pattern:
    """
    Represents a DAG using a func.func.
    The DAG has holes, represented by the function arguments.
    """

    func: FuncOp
    """The function representing the pattern."""
    semantics: FuncOp
    """The function representing the pattern's semantics."""

    size: int
    """
    The number of connectors in the pattern.
    In particular, operations that are CSE'd count every time they are used.
    """

    exact_fingerprint: bool
    """
    Wether the pattern's fingerprint represents exactly the pattern behavior.
    """
    evaluation_points_per_argument: tuple[tuple[Attribute, ...], ...]
    """
    The set of values used per argument to evaluate the pattern.
    """
    evaluation_points: tuple[tuple[Attribute, ...], ...]
    """
    The set of inputs on which the pattern is evaluated for the different fingerprints.
    """
    ordered_fingerprint: tuple[Result, ...]
    """
    The evaluation of the pattern on an ordered set of inputs.
    """
    unordered_fingerprint: UnorderedFingerprint
    """
    The multiset of results of the pattern on a set of inputs.
    """

    useless_parameters: frozenset[int]
    """
    The set of useless parameters, which do not have any effect in the computation.
    """

    def __init__(self, func: FuncOp, semantics: FuncOp) -> None:
        self.func = func
        self.semantics = semantics

        self.size = Pattern._formula_size(self.ret())

        values_for_each_param = [
            values_of_type(ty) for ty in self.semantics.function_type.inputs
        ]
        self.exact_fingerprint = all(total for _, total in values_for_each_param)

        self.useless_parameters = frozenset(self._compute_useless_parameters())

        # We don't need to compute multiple values for useless parameters.
        self.evaluation_points_per_argument = tuple(
            (
                (values_for_each_param[i][0][0],)
                if i in self.useless_parameters
                else tuple(values_for_each_param[i][0])
            )
            for i in range(self.semantics_arity)
        )
        interpreter = build_interpreter(ModuleOp([self.semantics.clone()]), 64)
        self.evaluation_points = Pattern.get_evaluation_points(
            self.evaluation_points_per_argument
        )
        self.ordered_fingerprint = tuple(
            interpret(interpreter, point) for point in self.evaluation_points
        )
        self.unordered_fingerprint = FrozenMultiset[tuple[Result, ...]].from_iterable(
            self.permutated_fingerprint(permutation)
            for permutation in self.input_permutations()
        )

    def ret(self) -> SSAValue:
        """Return the root value of the pattern."""
        return_op = self.func.get_return_op()
        assert return_op is not None
        return return_op.operands[0]

    @property
    def arity(self) -> int:
        """
        Returns the number of holes in the DAG, which corresponds to the number of
        function arguments.
        """
        return len(self.func.function_type.inputs)

    @property
    def semantics_arity(self) -> int:
        """
        Returns the number of arguments in the semantics function.
        This number might differ from `arity` when an additional state is passed to the function.
        """
        return len(self.semantics.function_type.inputs)

    def permutated_fingerprint(self, permutation: Permutation) -> tuple[Result, ...]:
        """
        Returns the pattern ordered fingerprint when its arguments are permuted.
        """
        evaluation_points_per_arguments = permute(
            self.evaluation_points_per_argument, permutation
        )
        evaluation_points = Pattern.get_evaluation_points(
            evaluation_points_per_arguments
        )
        mapping_fingerprint = {
            permute(point, permutation): value
            for point, value in zip(
                self.evaluation_points, self.ordered_fingerprint, strict=True
            )
        }

        return tuple(mapping_fingerprint[point] for point in evaluation_points)

    @staticmethod
    def get_evaluation_points(
        argument_points: tuple[tuple[Attribute, ...], ...],
    ) -> tuple[tuple[Attribute, ...], ...]:
        """Returns the evaluation points for the given argument points."""
        return tuple(
            inputs for inputs in itertools.product(*(vals for vals in argument_points))
        )

    @staticmethod
    def _formula_size(formula: SSAValue) -> int:
        """
        The size of a formula is defined as the number of connectors it contains.
        A connector is an operation that has at least one operand, so constants do
        not count as connectors.
        """
        match formula:
            case BlockArgument():
                return 0
            case OpResult(op=op):
                if len(op.operands) == 0:
                    return 0
                return 1 + sum(
                    Pattern._formula_size(operand) for operand in op.operands
                )
            case x:
                raise ValueError(f"Unknown value: {x}")

    def _is_parameter_useless(self, arg_index: int) -> bool:
        """
        Use Z3 to check wether if the argument at `arg_index` is irrelevant in
        the computation of the pattern.
        """

        module = ModuleOp([])
        builder = Builder(InsertPoint.at_end(module.body.block))

        # Clone the function twice into the new module
        func1 = builder.insert(clone_func_to_smt_func(self.semantics))
        func2 = builder.insert(func1.clone())
        function_type = func1.func_type

        # Declare one variable for each function input, and two for the input we
        # want to check.
        args1 = list[SSAValue]()
        args2 = list[SSAValue]()

        for i, arg_type in enumerate(function_type.inputs):
            arg1 = builder.insert(smt.DeclareConstOp(arg_type)).res
            args1.append(arg1)
            if i == arg_index:
                # We declare two variables for the argument we want to check.
                arg2 = builder.insert(smt.DeclareConstOp(arg_type)).res
                args2.append(arg2)
            else:
                args2.append(arg1)

        # Call the functions with their set of arguments
        call1 = builder.insert(smt.CallOp(func1.ret, args1)).res
        call2 = builder.insert(smt.CallOp(func2.ret, args2)).res

        # Check if the results are not equal.
        all_distinct = builder.insert(smt.ConstantBoolOp(False)).result
        for res1, res2 in zip(call1, call2, strict=True):
            res_distinct = builder.insert(smt.DistinctOp(res1, res2)).res
            all_distinct = builder.insert(smt.OrOp(all_distinct, res_distinct)).result

        builder.insert(smt.AssertOp(all_distinct))

        # Inline the function calls, and remove the function definitions.
        FunctionCallInline(True, {}).apply(Context(), module)
        for op in tuple(module.body.ops):
            if isinstance(op, smt.DefineFunOp):
                assert not op.ret.uses
                Rewriter().erase_op(op)

        return (
            z3.unsat == run_module_through_smtlib(module)[0]
        )  # pyright: ignore[reportUnknownVariableType]

    def _compute_useless_parameters(self) -> set[int]:
        """
        Compute the set of parameters that do not have any effect in the
        computation of the pattern.
        """
        values_for_each_param = [
            values_of_type(ty) for ty in self.semantics.function_type.inputs
        ]
        interpreter = build_interpreter(ModuleOp([self.semantics.clone()]), 64)

        # This list contains, for each parameter i, a map from the values of all
        # other parameters to the set of results obtained when varying parameter i.
        results_for_fixed_inputs: list[
            dict[tuple[Attribute, ...], set[tuple[Any, ...]]]
        ] = [{} for _ in range(self.semantics_arity)]

        for inputs in itertools.product(*(vals for vals, _ in values_for_each_param)):
            result = interpret(interpreter, inputs)
            for i in range(self.semantics_arity):
                results_for_fixed_inputs[i].setdefault(
                    inputs[:i] + inputs[i + 1 :], set()
                ).add(result)

        # A parameter is useless if, for every other parameters' values, varying
        # this parameter does not change the result.
        useless_parameters_on_cvec = {
            i
            for i, results_for_fixed_input in enumerate(results_for_fixed_inputs)
            if all(len(results) == 1 for results in results_for_fixed_input.values())
        }

        if self.exact_fingerprint:
            return useless_parameters_on_cvec

        # Then, compute which of those parameters are actually useless.
        return {i for i in useless_parameters_on_cvec if self._is_parameter_useless(i)}

    def input_permutations(self) -> Iterable[Permutation]:
        """
        Returns the permutation of all arguments that have an effect on the
        behavior of the pattern.
        """
        num_parameters_to_permute = self.arity - len(self.useless_parameters)
        permutation_index_to_parameter = [
            i for i in range(self.arity) if i not in self.useless_parameters
        ]
        parameter_to_permutation_index = {
            param: index for index, param in enumerate(permutation_index_to_parameter)
        }
        for permutation in itertools.permutations(range(num_parameters_to_permute)):
            yield tuple(
                i
                if i >= self.arity or self.useless_parameters
                else permutation_index_to_parameter[
                    permutation[parameter_to_permutation_index[i]]
                ]
                for i in range(self.semantics_arity)
            )

    def ordered_patterns(self) -> Iterable[OrderedPattern]:
        """
        Returns all the ordered patterns that are equivalent to this pattern,
        up to permutation of the arguments.
        """
        for permutation in self.input_permutations():
            yield OrderedPattern(
                self.func,
                self.semantics,
                permutation,
            )

    def is_same_behavior(self, other: Pattern) -> bool:
        """
        Tests whether two programs are logically equivalent up to parameter
        permutation, and ignoring useless parameters.
        """

        if self.unordered_fingerprint != other.unordered_fingerprint:
            return False

        if self.exact_fingerprint and other.exact_fingerprint:
            return True

        other_ordered = next(iter(other.ordered_patterns()))
        for permuted in self.ordered_patterns():
            if permuted.has_same_behavior(other_ordered):
                return True
        return False

    def permute_parameters_to_match(
        self, other: OrderedPattern
    ) -> OrderedPattern | None:
        """
        Returns an identical program with permuted parameters that is logically
        equivalent to the other program, ignoring useless arguments. If no such
        program exists, returns `None`.
        """

        if self.unordered_fingerprint != other.unordered_fingerprint:
            return None

        for permuted in self.ordered_patterns():
            if permuted.fingerprint == other.fingerprint:
                if (
                    self.exact_fingerprint and other.exact_fingerprint
                ) or permuted.has_same_behavior(other):
                    return permuted

        return None

    def is_refinement(self, other: Pattern) -> bool:
        """
        Tests whether this program refines another program up to parameter
        permutation, and ignoring useless parameters.
        """

        other_ordered = next(iter(other.ordered_patterns()))
        for permuted in self.ordered_patterns():
            if permuted.has_refinement(other_ordered):
                return True

        return False


class OrderedPattern(Pattern):
    """
    A pattern where the order of the arguments matters.
    """

    permutation: Permutation
    """
    The permutation of the arguments that this pattern represents.
    """

    def __init__(
        self, func: FuncOp, semantics: FuncOp, permutation: Permutation
    ) -> None:
        super().__init__(func, semantics)
        self.permutation = permutation

    @property
    def ordered_func(self):
        func = self.func.clone()
        for new_idx, (arg_idx, arg_type) in enumerate(self.useful_parameters()):
            old_arg = func.args[arg_idx]
            new_arg = func.body.block.insert_arg(arg_type, new_idx)
            old_arg.replace_by(new_arg)
            func.body.block.erase_arg(old_arg)
        return func

    @staticmethod
    def _compare_values_lexicographically(
        left: SSAValue,
        right: SSAValue,
        left_params: Permutation,
        right_params: Permutation,
    ) -> int:
        match left, right:
            # Order parameters.
            case BlockArgument(index=i), BlockArgument(index=j):
                return left_params[i] - right_params[j]
            # Favor operations over arguments.
            case OpResult(), BlockArgument():
                return -1
            case BlockArgument(), OpResult():
                return 1
            # Order operations.
            case OpResult(op=lop, index=i), OpResult(op=rop, index=j):
                # Favor operations with smaller arity.
                if len(lop.operands) != len(rop.operands):
                    return len(lop.operands) - len(rop.operands)
                # Choose an arbitrary result if they are different.
                if not isinstance(lop, type(rop)):
                    return -1 if lop.name < rop.name else 1
                if lop.properties != rop.properties:
                    sorted_keys = sorted(
                        set(lop.properties.keys() | rop.properties.keys())
                    )
                    for key in sorted_keys:
                        if key not in lop.properties:
                            return -1
                        if key not in rop.properties:
                            return 1
                        if lop.properties[key] != rop.properties[key]:
                            # Favor smaller properties.
                            return (
                                -1
                                if str(lop.properties[key]) < str(rop.properties[key])
                                else 1
                            )
                    assert False, "Logical error"
                if i != j:
                    return i - j
                # Compare operands as a last resort.
                for lo, ro in zip(lop.operands, rop.operands, strict=True):
                    c = OrderedPattern._compare_values_lexicographically(
                        lo, ro, left_params, right_params
                    )
                    if c != 0:
                        return c
                return 0
            case l, r:
                raise ValueError(f"Unknown value: {l} or {r}")

    def _compare_lexicographically(self, other: OrderedPattern) -> int:
        # Favor smaller programs.
        if self.size < other.size:
            return -1
        if self.size > other.size:
            return 1
        # Favor programs with fewer useless parameters.
        if len(self.useless_parameters) < len(other.useless_parameters):
            return -1
        if len(self.useless_parameters) > len(other.useless_parameters):
            return 1
        # Favor programs with fewer parameters overall.
        if self.arity < other.arity:
            return -1
        if self.arity > other.arity:
            return 1
        # Favor programs that return less values.
        self_out = self.ret()
        other_out = other.ret()
        return OrderedPattern._compare_values_lexicographically(
            self_out,
            other_out,
            self.permutation,
            other.permutation,
        )
        return 0

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, OrderedPattern):
            return NotImplemented
        return self._compare_lexicographically(other) < 0

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, OrderedPattern):
            return NotImplemented
        return self._compare_lexicographically(other) <= 0

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, OrderedPattern):
            return NotImplemented
        return self._compare_lexicographically(other) > 0

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, OrderedPattern):
            return NotImplemented
        return self._compare_lexicographically(other) >= 0

    @property
    def fingerprint(self) -> tuple[Result, ...]:
        """The ordered fingerprint of the pattern."""
        return self.permutated_fingerprint(self.permutation)

    def useful_parameters(self) -> Iterable[tuple[int, Attribute]]:
        """
        Returns the list of useful parameters in the pattern, in order.
        """
        useful_parameters = dict[int, Attribute]()
        for i, ty in enumerate(self.func.function_type.inputs):
            if i not in self.useless_parameters:
                useful_parameters[self.permutation[i]] = ty
        return useful_parameters.items()

    def useful_semantics_parameters(self) -> Iterable[tuple[int, Attribute]]:
        """
        Returns the list of useful parameters in the semantics function,
        as (index, type) pairs.
        """
        useful_parameters = dict[int, Attribute]()
        for i, ty in enumerate(self.semantics.function_type.inputs):
            if i not in self.useless_parameters:
                useful_parameters[self.permutation[i]] = ty
        return useful_parameters.items()

    def has_same_signature(self, other: OrderedPattern) -> bool:
        """
        Tests whether two patterns have the same signature, meaning the same
        holes given the order from `permutation`.
        """

        if len(self.func.function_type.inputs) != len(other.func.function_type.inputs):
            return False

        inputs = self.func.function_type.inputs.data
        other_inputs = other.func.function_type.inputs.data

        ordered_inputs = permute(inputs, self.permutation[: len(inputs)])
        other_ordered_inputs = permute(
            other_inputs, other.permutation[: len(other_inputs)]
        )

        return ordered_inputs == other_ordered_inputs

    def has_same_behavior(self, other: OrderedPattern) -> bool:
        """Tests whether two programs are logically equivalent."""

        if not self.has_same_signature(other):
            return False

        if self.fingerprint != other.fingerprint:
            return False

        if self.exact_fingerprint and other.exact_fingerprint:
            return True

        return self._is_same_behavior_with_z3(other)

    def _is_same_behavior_with_z3(
        self,
        other: OrderedPattern,
    ) -> bool:
        """
        Check wether two programs are semantically equivalent after permuting the
        arguments of the left program, using Z3. This also checks whether the input
        types match (after permutation and removal of useless parameters).
        """

        module = ModuleOp([])
        builder = Builder(InsertPoint.at_end(module.body.block))

        # Clone both functions into a new module.
        func_self = builder.insert(clone_func_to_smt_func(self.semantics))
        func_other = builder.insert(clone_func_to_smt_func(other.semantics))

        # Declare a variable for each function input.
        args_self: list[SSAValue | None] = [None] * len(func_self.func_type.inputs)
        args_other: list[SSAValue | None] = [None] * len(func_other.func_type.inputs)
        for (self_index, self_type), (other_index, other_type) in zip(
            self.useful_semantics_parameters(), other.useful_semantics_parameters()
        ):
            if self_type != other_type:
                return False
            arg = builder.insert(smt.DeclareConstOp(self_type)).res
            args_self[self_index] = arg
            args_other[other_index] = arg

        for index, ty in enumerate(func_self.func_type.inputs):
            if args_self[index] is None:
                args_self[index] = builder.insert(smt.DeclareConstOp(ty)).res

        for index, ty in enumerate(func_other.func_type.inputs):
            if args_other[index] is None:
                args_other[index] = builder.insert(smt.DeclareConstOp(ty)).res

        args_self_complete = cast(list[SSAValue], args_self)
        args_other_complete = cast(list[SSAValue], args_other)

        # Call each function with the same arguments.
        self_call = builder.insert(smt.CallOp(func_self.ret, args_self_complete)).res
        other_call = builder.insert(smt.CallOp(func_other.ret, args_other_complete)).res

        # Check if the two results are not equal.
        check = builder.insert(smt.ConstantBoolOp(False)).result
        for res1, res2 in zip(self_call, other_call, strict=True):
            # Hack to handle poison equality.
            if isa(res1.type, pair.PairType[bv.BitVectorType, smt.BoolType]):
                value1 = builder.insert(pair.FirstOp(res1)).res
                value2 = builder.insert(pair.FirstOp(res2)).res
                poison1 = builder.insert(pair.SecondOp(res1)).res
                poison2 = builder.insert(pair.SecondOp(res2)).res
                # They are distinct if (poison1 != poison2) or (poison1 == poison2 == false and value1 != value2)
                poison_distinct = builder.insert(smt.DistinctOp(poison1, poison2)).res
                value_distinct = builder.insert(smt.DistinctOp(value1, value2)).res
                both_not_poison = builder.insert(
                    smt.AndOp(
                        builder.insert(smt.NotOp(poison1)).result,
                        builder.insert(smt.NotOp(poison2)).result,
                    )
                ).result
                distinct = builder.insert(
                    smt.OrOp(
                        poison_distinct,
                        builder.insert(
                            smt.AndOp(both_not_poison, value_distinct)
                        ).result,
                    )
                ).result
            else:
                distinct = builder.insert(smt.DistinctOp(res1, res2)).res
            distinct.name_hint = "result_different"
            check = builder.insert(smt.OrOp(check, distinct)).result
        builder.insert(smt.AssertOp(check))

        FunctionCallInline(True, {}).apply(Context(), module)
        for op in tuple(module.body.ops):
            if isinstance(op, smt.DefineFunOp):
                assert not op.ret.uses
                Rewriter().erase_op(op)

        # Now that we have the module, run it through the Z3 solver.
        return (
            z3.unsat == run_module_through_smtlib(module)[0]
        )  # pyright: ignore[reportUnknownVariableType]

    def has_refinement(self, other: OrderedPattern) -> bool:
        """Tests whether a program refines an order program."""

        if not self.has_same_signature(other):
            return False

        # if self.fingerprint != other.fingerprint:
        #     return False

        # if self.exact_fingerprint and other.exact_fingerprint:
        #     return True

        return self._is_refinement_with_z3(other)

    def _is_refinement_with_z3(
        self,
        other: OrderedPattern,
    ) -> bool:
        """
        Check wether this program refines another program after permuting the
        arguments of the left program, using Z3. This also checks whether the input
        types match (after permutation and removal of useless parameters).
        """

        module = ModuleOp([])
        builder = Builder(InsertPoint.at_end(module.body.block))

        # Clone both functions into a new module.
        func_self = builder.insert(clone_func_to_smt_func(self.semantics))
        func_other = builder.insert(clone_func_to_smt_func(other.semantics))

        # Declare a variable for each function input.
        args_self: list[SSAValue | None] = [None] * len(func_self.func_type.inputs)
        args_other: list[SSAValue | None] = [None] * len(func_other.func_type.inputs)
        for (self_index, self_type), (other_index, other_type) in zip(
            self.useful_semantics_parameters(), other.useful_semantics_parameters()
        ):
            if self_type != other_type:
                return False
            arg = builder.insert(smt.DeclareConstOp(self_type)).res
            args_self[self_index] = arg
            args_other[other_index] = arg

        for index, ty in enumerate(func_self.func_type.inputs):
            if args_self[index] is None:
                args_self[index] = builder.insert(smt.DeclareConstOp(ty)).res

        for index, ty in enumerate(func_other.func_type.inputs):
            if args_other[index] is None:
                args_other[index] = builder.insert(smt.DeclareConstOp(ty)).res

        args_self_complete = cast(list[SSAValue], args_self)
        args_other_complete = cast(list[SSAValue], args_other)

        # Call each function with the same arguments.
        self_call = builder.insert(smt.CallOp(func_self.ret, args_self_complete))
        other_call = builder.insert(smt.CallOp(func_other.ret, args_other_complete))

        refinement = function_results_refinement(
            other_call,
            other.func.function_type,
            self_call,
            self.func.function_type,
            builder.insertion_point,
        )
        not_refinement = builder.insert(smt.NotOp(refinement)).result
        builder.insert(smt.AssertOp(not_refinement))
        builder.insert

        FunctionCallInline(True, {}).apply(Context(), module)
        for op in tuple(module.body.ops):
            if isinstance(op, smt.DefineFunOp):
                assert not op.ret.uses
                Rewriter().erase_op(op)

        # Now that we have the module, run it through the Z3 solver.
        return (
            z3.unsat == run_module_through_smtlib(module)[0]
        )  # pyright: ignore[reportUnknownVariableType]
