"""
This file defines the Pattern class, which represent a DAG with holes using a
function. The Pattern class additionally stores the semantics of the DAG, as
well as additional semantic information.

"""

from dataclasses import dataclass
from typing import Any, cast, Sequence, TypeVar, Iterable
import itertools
import z3  # pyright: ignore[reportMissingTypeStubs]

from xdsl.ir import SSAValue, BlockArgument, OpResult, Attribute
from xdsl.context import Context
from xdsl.builder import Builder
from xdsl.rewriter import InsertPoint, Rewriter

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


Permutation = tuple[int, ...]

T = TypeVar("T")


def permute(seq: Sequence[T], permutation: Permutation) -> tuple[T, ...]:
    assert len(seq) == len(permutation)
    return tuple(seq[i] for i in permutation)


def reverse_permute(seq: Sequence[T], permutation: Permutation) -> tuple[T, ...]:
    assert len(seq) == len(permutation)
    result: list[T | None] = [None for _ in seq]
    for x, i in zip(seq, permutation, strict=True):
        result[i] = x
    assert all(x is not None for x in result)
    return tuple(cast(list[T], result))


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


@dataclass(slots=True, eq=False)
class Fingerprint:
    """
    The evaluation of a program on a set of inputs. It is composed of an
    ordered fingerprint, and an unordered fingerprint.

    Two patterns that are semantically equivalent up to permutation will have
    the same unordered fingerprint. When the patterns are semantically equivalent
    (not just up to permutation), they will also have the same ordered fingerprint.
    """

    ordered: tuple[Result, ...]
    """
    The evaluation of the program on an ordered set of inputs.
    """
    unordered: FrozenMultiset[tuple[Result, ...]]
    """
    The multiset of results of the program on a set of inputs.
    """
    mapping: dict[tuple[Attribute, ...], Result]
    """
    The evaluation of the program on a set of inputs, indexed by the inputs.
    """


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
    unordered_fingerprint: FrozenMultiset[tuple[Result, ...]]
    """
    The multiset of results of the pattern on a set of inputs.
    """
    mapping_fingerprint: dict[tuple[Attribute, ...], Result]
    """
    The evaluation of the pattern on a set of inputs, indexed by the inputs.
    """

    useless_parameters: set[int]
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

        self.useless_parameters = self._compute_useless_parameters()

        # We don't need to compute multiple values for useless parameters.
        self.evaluation_points_per_argument = tuple(
            (
                (values_for_each_param[i][0][0],)
                if i in self.useless_parameters
                else tuple(values_for_each_param[i][0])
            )
            for i in range(self.arity + 1)
        )
        interpreter = build_interpreter(ModuleOp([self.semantics.clone()]), 64)
        self.evaluation_points = Pattern.get_evaluation_points(
            self.evaluation_points_per_argument
        )
        self.ordered_fingerprint = tuple(
            interpret(interpreter, point) for point in self.evaluation_points
        )
        self.mapping_fingerprint = {
            point: value
            for point, value in zip(
                self.evaluation_points, self.ordered_fingerprint, strict=True
            )
        }
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
        return tuple(self.mapping_fingerprint[point] for point in evaluation_points)

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

        return z3.unsat == run_module_through_smtlib(
            module
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
        ] = [{} for _ in range(self.arity)]

        for inputs in itertools.product(*(vals for vals, _ in values_for_each_param)):
            result = interpret(interpreter, inputs)
            for i in range(self.arity):
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
                if i == self.arity or self.useless_parameters
                else permutation_index_to_parameter[
                    permutation[parameter_to_permutation_index[i]]
                ]
                for i in range(self.arity + 1)
            )
