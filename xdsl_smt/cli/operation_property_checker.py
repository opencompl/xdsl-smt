from xdsl_smt.dialects.smt_dialect import ConstantBoolOp, YieldOp, ForallOp
from xdsl.dialects.func import FuncOp
import argparse
import subprocess

from xdsl.context import MLContext
from xdsl.parser import Parser

from io import StringIO

from xdsl.utils.hints import isa
from ..dialects.smt_dialect import (
    SMTDialect,
    DefineFunOp,
    EqOp,
    AssertOp,
    CheckSatOp,
    DistinctOp,
)
from ..dialects.smt_bitvector_dialect import (
    SMTBitVectorDialect,
    ConstantOp,
)
from xdsl_smt.dialects.transfer import (
    AbstractValueType,
    TransIntegerType,
    TupleType,
)
from ..dialects.index_dialect import Index
from ..dialects.smt_utils_dialect import SMTUtilsDialect, FirstOp
from xdsl.ir import Block, Region, SSAValue
from xdsl.dialects.builtin import (
    Builtin,
    ModuleOp,
    IntegerAttr,
    IntegerType,
    i1,
    FunctionType,
    ArrayAttr,
    StringAttr,
    AnyArrayAttr,
)
from xdsl.dialects.func import Func, FuncOp, Return
from ..dialects.transfer import Transfer
from xdsl.dialects.arith import Arith
from xdsl.dialects.comb import Comb
from xdsl.dialects.hw import HW
from ..passes.dead_code_elimination import DeadCodeElimination
from ..passes.merge_func_results import MergeFuncResultsPass
from ..passes.transfer_inline import FunctionCallInline
import xdsl.dialects.comb as comb
from xdsl.ir import Operation
from ..passes.lower_to_smt.lower_to_smt import LowerToSMTPass, SMTLowerer
from ..passes.lower_effects import LowerEffectPass
from ..passes.lower_to_smt import (
    func_to_smt_patterns,
)
from ..utils.transfer_function_util import (
    SMTTransferFunction,
    FunctionCollection,
    TransferFunction,
    fixDefiningOpReturnType,
    getArgumentWidthsWithEffect,
    getArgumentInstancesWithEffect,
    callFunctionWithEffect,
    insertArgumentInstancesToBlockWithEffect,
)

from ..utils.transfer_function_check_util import (
    forward_soundness_check,
    backward_soundness_check,
    counterexample_check,
    int_attr_check,
    forward_precision_check,
    module_op_validity_check,
)
from ..passes.transfer_unroll_loop import UnrollTransferLoop
from xdsl_smt.semantics import transfer_semantics
from ..traits.smt_printer import print_to_smtlib
from xdsl_smt.passes.lower_pairs import LowerPairs
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl_smt.semantics.arith_semantics import arith_semantics
from xdsl_smt.semantics.builtin_semantics import IntegerTypeSemantics
from xdsl_smt.semantics.transfer_semantics import (
    transfer_semantics,
    AbstractValueTypeSemantics,
    TransferIntegerTypeSemantics,
)
from xdsl_smt.semantics.comb_semantics import comb_semantics
import sys as sys


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument(
        "transfer_functions", type=str, nargs="?", help="path to the transfer functions"
    )


def verify_pattern(ctx: MLContext, op: ModuleOp) -> bool:
    cloned_op = op.clone()
    stream = StringIO()
    LowerPairs().apply(ctx, cloned_op)
    CanonicalizePass().apply(ctx, cloned_op)
    DeadCodeElimination().apply(ctx, cloned_op)

    print_to_smtlib(cloned_op, stream)
    # print(stream.getvalue())
    res = subprocess.run(
        ["z3", "-in"],
        capture_output=True,
        input=stream.getvalue(),
        text=True,
    )
    if res.returncode != 0:
        raise Exception(res.stderr)
    return "unsat" in res.stdout


def get_model(ctx: MLContext, op: ModuleOp) -> tuple[bool, str]:
    cloned_op = op.clone()
    stream = StringIO()
    LowerPairs().apply(ctx, cloned_op)
    CanonicalizePass().apply(ctx, cloned_op)
    DeadCodeElimination().apply(ctx, cloned_op)

    print_to_smtlib(cloned_op, stream)
    print("\n(eval const_first)\n", file=stream)
    # print(stream.getvalue())
    res = subprocess.run(
        ["z3", "-in"],
        capture_output=True,
        input=stream.getvalue(),
        text=True,
    )
    if res.returncode != 0:
        return False, ""
    return True, str(res.stdout)


def lowerToSMTModule(module: ModuleOp, width: int, ctx: MLContext):
    # lower to SMT
    SMTLowerer.rewrite_patterns = {
        **func_to_smt_patterns,
    }
    SMTLowerer.type_lowerers = {
        IntegerType: IntegerTypeSemantics(),
        AbstractValueType: AbstractValueTypeSemantics(),
        TransIntegerType: TransferIntegerTypeSemantics(width),
        # tuple and abstract use the same type lowerers
        TupleType: AbstractValueTypeSemantics(),
    }
    SMTLowerer.op_semantics = {
        **arith_semantics,
        **transfer_semantics,
        **comb_semantics,
    }
    LowerToSMTPass().apply(ctx, module)
    MergeFuncResultsPass().apply(ctx, module)
    LowerEffectPass().apply(ctx, module)


def parse_file(ctx: MLContext, file: str | None) -> Operation:
    if file is None:
        f = sys.stdin
        file = "<stdin>"
    else:
        f = open(file)

    parser = Parser(ctx, f.read(), file)
    module = parser.parse_op()
    return module


def get_int_val_from_smt_return(result_str: str) -> str:
    lines = result_str.split("\n")
    for line in lines:
        if line.startswith("#x"):
            int_val = int(line.replace("#", "0"), 16)
            return str(int_val)
    assert False


class OperationProperties:
    operation: Operation
    commutativity: bool
    associativity: bool
    involution: bool
    idempotence: bool
    absorbing_elements: list[list[str]]
    identity_elements: list[list[str]]
    self_annihilation_elements: list[str]
    distribution_operations: list[Operation]

    def __init__(self, operation: Operation):
        self.operation = operation
        self.absorbing_elements = []
        self.self_annihilation_elements = []
        self.identity_elements = []


class PropertyPrinter:
    operation: OperationProperties
    indent: str
    constant_cnt: int

    def __init__(self, operation: OperationProperties):
        self.constant_cnt = 0
        self.operation = operation
        self.indent = ""

    def get_constant_name(self) -> str:
        result = "const" + str(self.constant_cnt)
        self.constant_cnt += 1
        return result

    def get_nth_operand(self, idx: int) -> str:
        nth_operand = "op.getOperand({idx})"
        return nth_operand.format(idx=idx)

    def get_constant(self, val: str) -> tuple[str, str]:
        width = (
            self.indent
            + "unsigned width = op.getOperand(0).getType().getIntOrFloatBitWidth();\n"
        )
        result = self.indent + "llvm::APInt result(width, {val});\n"
        return width + result.format(val=val), "result"

    def match_constant(self, val: str, const_name: str) -> str:
        return "matchPattern({val}, m_ConstantInt(&{const_name}))".format(
            val=val, const_name=const_name
        )

    def match_constant_block(
        self,
    ) -> tuple[str, str]:
        constant_name = self.get_constant_name()
        apint_decl = self.indent + "llvm::APInt " + constant_name + ";\n"
        if_block = self.indent + ("if({cond}){{\n" "{content}" + self.indent + "}}\n")
        return apint_decl + if_block, constant_name

    def get_op_name(self) -> str:
        return self.operation.operation.name.replace(".", "_")

    def get_op_cpp_class(self) -> str:
        assert "CPPCLASS" in self.operation.operation.attributes
        cpp_class = self.operation.operation.attributes["CPPCLASS"].data[0]
        return cpp_class.data

    def push_indent(self):
        self.indent += "\t"

    def pop_indent(self):
        self.indent = self.indent[:-1]

    def generate_isa_cpp_class(self) -> str:
        if_block = self.indent + ("if({cond}){{\n" "{content}" + self.indent + "}}\n")
        cond = "llvm::isa<{cpp_class}>(op)"
        self.push_indent()
        content = ""
        content += self.generate_idempotence()
        content += self.generate_self_annihilation()
        content += self.generate_absorbing_elements()
        content += self.generate_identity_elements()
        self.pop_indent()
        return if_block.format(
            cond=cond.format(cpp_class=self.get_op_cpp_class()), content=content
        )

    def generate_idempotence(self) -> str:
        idempotence = self.operation.idempotence
        result = ""
        if idempotence:
            firstOperand: str = self.get_nth_operand(0)
            secondOperand: str = self.get_nth_operand(1)
            if_block = (
                self.indent + "if({cond}){{\n" "{return_val}" + self.indent + "}}\n"
            )
            self.push_indent()
            return_val = (self.indent + "return {constant_name};\n").format(
                constant_name=firstOperand
            )
            self.pop_indent()
            cond = "{firstOperand} == {secondOperand}"

            result += if_block.format(
                cond=cond.format(
                    firstOperand=firstOperand, secondOperand=secondOperand
                ),
                return_val=return_val,
            )

        return result

    def generate_self_annihilation(self) -> str:
        self_annihilation_elements = self.operation.self_annihilation_elements
        if len(self_annihilation_elements) != 0:
            result = ""
            for ele in self_annihilation_elements:
                firstOperand: str = self.get_nth_operand(0)
                secondOperand: str = self.get_nth_operand(1)
                if_block = (
                    self.indent + "if({cond}){{\n"
                    "{constant}\n"
                    "{return_val}" + self.indent + "}}\n"
                )
                self.push_indent()
                constant, constant_name = self.get_constant(ele)
                return_val = (
                    self.indent + "return getIntAttr({constant_name}, context);\n"
                ).format(constant_name=constant_name)
                self.pop_indent()
                cond = "{firstOperand} == {secondOperand}"

                result += if_block.format(
                    cond=cond.format(
                        firstOperand=firstOperand, secondOperand=secondOperand
                    ),
                    constant=constant,
                    return_val=return_val,
                )
            return result
        return ""

    def generate_absorbing_elements(self) -> str:
        absorbing_element = self.operation.absorbing_elements
        result = ""
        for i in range(len(absorbing_element)):
            for ele in absorbing_element[i]:
                constant_if_block, apint_name = self.match_constant_block()
                first_operand = self.get_nth_operand(i)
                first_cond = self.match_constant(first_operand, apint_name)
                self.push_indent()
                absorbing_str, absorbing_element = self.get_constant(ele)
                if_block = (
                    self.indent + "if({cond}){{\n" "{content}" + self.indent + "}}\n"
                )
                self.push_indent()
                return_val = (
                    self.indent
                    + "return getIntAttr("
                    + absorbing_element
                    + ", context);\n"
                )
                self.pop_indent()
                first_content = absorbing_str + if_block.format(
                    cond=apint_name + " == " + absorbing_element, content=return_val
                )
                self.pop_indent()
                result += constant_if_block.format(
                    cond=first_cond, content=first_content
                )
        return result

    def generate_identity_elements(self) -> str:
        identity_elements = self.operation.identity_elements
        result = ""
        for i in range(len(identity_elements)):
            for ele in identity_elements[i]:
                constant_if_block, apint_name = self.match_constant_block()
                first_operand = self.get_nth_operand(i)
                first_cond = self.match_constant(first_operand, apint_name)
                self.push_indent()
                identity_str, identity_elements = self.get_constant(ele)
                if_block = (
                    self.indent + "if({cond}){{\n" "{content}" + self.indent + "}}\n"
                )
                self.push_indent()
                return_val = self.indent + "return " + first_operand + ";\n"
                self.pop_indent()
                first_content = identity_str + if_block.format(
                    cond=apint_name + " == " + identity_elements, content=return_val
                )
                self.pop_indent()
                result += constant_if_block.format(
                    cond=first_cond, content=first_content
                )
        return result

    def generate_interface(self):
        op_name = self.get_op_name()
        head = (
            "static mlir::OpFoldResult "
            + op_name
            + "Simplify(mlir::Operation op, mlir::MLIRContext* context){\n"
        )
        self.push_indent()
        mid = self.generate_isa_cpp_class()
        end = self.indent + "return {};\n"
        self.pop_indent()
        end += self.indent + "}\n"
        return head + mid + end


"""
opp=OperationProperties()
opp.self_annihilation_elements=["0"]
opp.absorbing_elements=["0"]
opp.identity_elements=["0"]
printer=PropertyPrinter()
printer.operation=opp
printer.indent=""
print(printer.generate_interface())
exit(0)
"""


def check_commutativity(func: DefineFunOp, ctx: MLContext) -> bool:
    if len(func.body.block.args) == 3:
        query_module = ModuleOp([])
        # getInstance
        effect = ConstantBoolOp(False)
        arg_ops = getArgumentInstancesWithEffect(func, {})
        args = [arg.res for arg in arg_ops]

        # func(x, y)
        call_func_x_y_op, call_func_x_y_first_op = callFunctionWithEffect(
            func, args, effect.res
        )

        # func(y, x)
        call_func_y_x_op, call_func_y_x_first_op = callFunctionWithEffect(
            func, [args[1], args[0]], effect.res
        )

        # assert func(y, x) != func(x,y)
        call_distinct_op = DistinctOp(
            call_func_x_y_first_op.res, call_func_y_x_first_op.res
        )
        assert_op = AssertOp(call_distinct_op.res)

        all_ops = (
            arg_ops
            + [call_func_x_y_op, call_func_x_y_first_op]
            + [call_func_y_x_op, call_func_y_x_first_op]
            + [call_distinct_op, assert_op, CheckSatOp()]
        )

        query_module.body.block.add_ops(all_ops)
        FunctionCallInline(True, {}).apply(ctx, query_module)

        result = verify_pattern(ctx, query_module)
        print("Commutativity Check result:", result)
        return result
    return False


def check_associativity(func: DefineFunOp, ctx: MLContext) -> bool:
    if len(func.body.block.args) == 3:
        query_module = ModuleOp([])
        # getInstance x, y, z, useless
        effect = ConstantBoolOp(False)
        arg_ops = getArgumentInstancesWithEffect(func, {})
        arg_ops_again = getArgumentInstancesWithEffect(func, {})
        args = [arg.res for arg in arg_ops] + [arg.res for arg in arg_ops_again]

        # func(x, y)
        call_func_x_y_op, call_func_x_y_first_op = callFunctionWithEffect(
            func, [args[0], args[1]], effect.res
        )
        # func(func(x, y), z)
        call_func_func_x_y_z_op, call_func_func_x_y_z_first_op = callFunctionWithEffect(
            func, [call_func_x_y_first_op.res, args[2]], effect.res
        )

        # func(y, z)
        call_func_y_z_op, call_func_y_z_first_op = callFunctionWithEffect(
            func, [args[1], args[2]], effect.res
        )
        # func(x, func(y, z)
        call_func_x_func_y_z_op, call_func_x_func_y_z_first_op = callFunctionWithEffect(
            func, [args[0], call_func_y_z_first_op.res], effect.res
        )

        # assert func(func(x, y), z) == func(x,func(y, z))
        call_distinct_op = DistinctOp(
            call_func_func_x_y_z_first_op.res, call_func_x_func_y_z_first_op.res
        )
        assert_op = AssertOp(call_distinct_op.res)

        all_ops = (
            arg_ops
            + arg_ops_again
            + [call_func_x_y_op, call_func_x_y_first_op]
            + [call_func_func_x_y_z_op, call_func_func_x_y_z_first_op]
            + [call_func_y_z_op, call_func_y_z_first_op]
            + [call_func_x_func_y_z_op, call_func_x_func_y_z_first_op]
            + [call_distinct_op, assert_op, CheckSatOp()]
        )

        query_module.body.block.add_ops(all_ops)
        FunctionCallInline(True, {}).apply(ctx, query_module)

        result = verify_pattern(ctx, query_module)
        print("Associativity Check result:", result)
        return result
    return False


def check_involution(func: DefineFunOp, ctx: MLContext) -> bool:
    if len(func.body.block.args) == 2:
        query_module = ModuleOp([])
        # getInstance x
        effect = ConstantBoolOp(False)
        arg_ops = getArgumentInstancesWithEffect(func, {})
        args = [arg.res for arg in arg_ops]
        # func(func(x)) == x

        # func(x)
        call_func_x_op, call_func_x_first_op = callFunctionWithEffect(
            func, args, effect.res
        )
        # func(func(x))
        call_func_func_x_op, call_func_func_x_first_op = callFunctionWithEffect(
            func, [call_func_x_first_op.res], effect.res
        )

        # assert func(func(x)) != x
        call_distinct_op = DistinctOp(call_func_func_x_first_op.res, args[0])
        assert_op = AssertOp(call_distinct_op.res)

        all_ops = (
            arg_ops
            + [call_func_x_op, call_func_x_first_op]
            + [call_func_func_x_op, call_func_func_x_first_op]
            + [call_distinct_op, assert_op, CheckSatOp()]
        )

        query_module.body.block.add_ops(all_ops)
        FunctionCallInline(True, {}).apply(ctx, query_module)

        result = verify_pattern(ctx, query_module)
        print("Involution Check result:", result)
        return result
    return False


def check_idempotence(func: DefineFunOp, ctx: MLContext) -> bool:
    # func(x, x) == x
    if len(func.body.block.args) == 3:
        query_module = ModuleOp([])
        # getInstance x, y
        effect = ConstantBoolOp(False)
        arg_ops = getArgumentInstancesWithEffect(func, {})
        args = [arg.res for arg in arg_ops]

        # assert x == y
        x_eq_y_op = EqOp(args[0], args[1])
        assert_x_eq_y_op = AssertOp(x_eq_y_op.res)

        # func(x, x)
        call_func_x_x_op, call_func_x_x_first_op = callFunctionWithEffect(
            func, args, effect.res
        )

        # assert func(y, x) != x
        call_distinct_op = DistinctOp(call_func_x_x_first_op.res, args[0])
        assert_op = AssertOp(call_distinct_op.res)

        all_ops = (
            arg_ops
            + [x_eq_y_op, assert_x_eq_y_op]
            + [call_func_x_x_op, call_func_x_x_first_op]
            + [call_distinct_op, assert_op, CheckSatOp()]
        )

        query_module.body.block.add_ops(all_ops)
        FunctionCallInline(True, {}).apply(ctx, query_module)

        result = verify_pattern(ctx, query_module)
        print("Idempotence Check result:", result)
        return result
    return False


# Forall([x], op(x,ele)==ele)
def get_forall_absorbing_property(
    op: DefineFunOp, ele: SSAValue, is_first_operand: bool, effect: ConstantBoolOp
) -> ForallOp:
    forall_block = Block()
    forall_block_args = insertArgumentInstancesToBlockWithEffect(op, {}, forall_block)

    if is_first_operand:
        op_args = [ele, forall_block_args[0]]
    else:
        op_args = [forall_block_args[0], ele]

    # func(x, ele)
    call_func_x_ele_op, call_func_x_ele_first_op = callFunctionWithEffect(
        op, op_args, effect.res
    )

    call_func_x_ele_first_first_op = FirstOp(call_func_x_ele_first_op.res)
    ele_first_op = FirstOp(ele)

    # func(x,ele)==ele
    call_func_x_ele_eq_ele_op = EqOp(
        call_func_x_ele_first_first_op.res, ele_first_op.res
    )

    yield_op = YieldOp(call_func_x_ele_eq_ele_op.res)

    forall_all_ops = (
        [call_func_x_ele_op, call_func_x_ele_first_op]
        + [call_func_x_ele_first_first_op, ele_first_op]
        + [
            call_func_x_ele_eq_ele_op,
            yield_op,
        ]
    )
    forall_block.add_ops(forall_all_ops)

    forall_op = ForallOp.from_variables([], Region(forall_block))
    return forall_op


# Forall([x], op(x,ele)==ele)
# Forall([x], op(ele,x)==ele)
def check_absorbing_element(
    func: DefineFunOp, ctx: MLContext, commutativity: bool
) -> list[list[str]]:
    result_lst: list[list[str]] = []
    if len(func.body.block.args) == 3:
        operand_order = [True]
        if not commutativity:
            operand_order.append(False)

        final_result = False
        for op_order in operand_order:
            result_lst.append([])
            query_module = ModuleOp([])
            # getInstance x, y
            effect = ConstantBoolOp(False)
            arg_ops = getArgumentInstancesWithEffect(func, {})
            args = [arg.res for arg in arg_ops]

            forall_op = get_forall_absorbing_property(func, args[0], op_order, effect)

            # assert forall
            assert_forall_op = AssertOp(forall_op.res)

            all_ops = arg_ops + [forall_op, assert_forall_op] + [CheckSatOp()]

            query_module.body.block.add_ops(all_ops)
            FunctionCallInline(True, {}).apply(ctx, query_module)

            result, result_str = get_model(ctx, query_module)

            def print_absorbing_element(
                result: bool, func_name: str, s: str, is_first_operand: bool
            ) -> str:
                if result:
                    int_val = get_int_val_from_smt_return(s)
                    return "{func_name}({first_op}, {second_op}) == {int_val}".format(
                        func_name=func_name,
                        first_op=(int_val if is_first_operand else "x"),
                        second_op=(int_val if not is_first_operand else "x"),
                        int_val=int_val,
                    )

                return "N/A"

            print(
                "Absorbing Element Check result: ",
                print_absorbing_element(
                    result, func.fun_name.data, result_str, op_order
                ),
            )
            final_result |= result
            if result:
                result_lst[-1].append(get_int_val_from_smt_return(result_str))

    return result_lst


# Forall([x], op(x,ele)==x)
def get_forall_identity_property(
    op: DefineFunOp, ele: SSAValue, is_first_operand: bool, effect: ConstantBoolOp
) -> ForallOp:
    forall_block = Block()
    forall_block_args = insertArgumentInstancesToBlockWithEffect(op, {}, forall_block)

    if is_first_operand:
        op_args = [ele, forall_block_args[0]]
    else:
        op_args = [forall_block_args[0], ele]

    # func(x, ele)
    call_func_x_ele_op, call_func_x_ele_first_op = callFunctionWithEffect(
        op, op_args, effect.res
    )

    # remove poison
    call_func_x_ele_first_first_op = FirstOp(call_func_x_ele_first_op.res)
    x_first_op = FirstOp(forall_block_args[0])

    # func(x,ele)==x
    call_func_x_ele_eq_x_op = EqOp(call_func_x_ele_first_first_op.res, x_first_op.res)

    yield_op = YieldOp(call_func_x_ele_eq_x_op.res)

    forall_all_ops = (
        [call_func_x_ele_op, call_func_x_ele_first_op]
        + [call_func_x_ele_first_first_op, x_first_op]
        + [
            call_func_x_ele_eq_x_op,
            yield_op,
        ]
    )
    forall_block.add_ops(forall_all_ops)

    forall_op = ForallOp.from_variables([], Region(forall_block))
    return forall_op


# Forall([x], op(x,ele)==x)
# Forall([x], op(ele,x)==x)
def check_identity_element(
    func: DefineFunOp, ctx: MLContext, commutativity: bool
) -> list[list[str]]:
    result_lst: list[list[str]] = []
    if len(func.body.block.args) == 3:
        operand_order = [True]
        if not commutativity:
            operand_order.append(False)

        final_result = False
        for op_order in operand_order:
            result_lst.append([])
            query_module = ModuleOp([])
            # getInstance x, y
            effect = ConstantBoolOp(False)
            arg_ops = getArgumentInstancesWithEffect(func, {})
            args = [arg.res for arg in arg_ops]

            forall_op = get_forall_identity_property(func, args[0], op_order, effect)

            # assert forall
            assert_forall_op = AssertOp(forall_op.res)

            all_ops = arg_ops + [forall_op, assert_forall_op] + [CheckSatOp()]

            query_module.body.block.add_ops(all_ops)
            FunctionCallInline(True, {}).apply(ctx, query_module)

            result, result_str = get_model(ctx, query_module)

            def print_zero_element(
                result: bool, func_name: str, s: str, is_first_operand: bool
            ) -> str:
                if result:
                    int_val = get_int_val_from_smt_return(s)
                    return "{func_name}({first_op}, {second_op}) == x".format(
                        func_name=func_name,
                        first_op=(int_val if is_first_operand else "x"),
                        second_op=(int_val if not is_first_operand else "x"),
                    )

                return "N/A"

            print(
                "Identity Element Check result: ",
                print_zero_element(result, func.fun_name.data, result_str, op_order),
            )
            final_result |= result
            if result:
                result_lst[-1].append(get_int_val_from_smt_return(result_str))
    return result_lst


# Forall([x], op(x,x)==ele)
def get_forall_self_annihilation_property(
    op: DefineFunOp, ele: SSAValue, effect: ConstantBoolOp
) -> ForallOp:
    forall_block = Block()
    forall_block_args = insertArgumentInstancesToBlockWithEffect(op, {}, forall_block)

    # func(x, x)
    call_func_x_x_op, call_func_x_x_first_op = callFunctionWithEffect(
        op, [forall_block_args[0], forall_block_args[0]], effect.res
    )

    call_func_x_x_first_first_op = FirstOp(call_func_x_x_first_op.res)
    ele_first_op = FirstOp(ele)

    # func(x,x)==ele
    call_func_x_x_eq_ele_op = EqOp(call_func_x_x_first_first_op.res, ele_first_op.res)

    yield_op = YieldOp(call_func_x_x_eq_ele_op.res)

    forall_all_ops = (
        [call_func_x_x_op, call_func_x_x_first_op]
        + [call_func_x_x_first_first_op, ele_first_op]
        + [
            call_func_x_x_eq_ele_op,
            yield_op,
        ]
    )
    forall_block.add_ops(forall_all_ops)

    forall_op = ForallOp.from_variables([], Region(forall_block))
    return forall_op


def check_self_annihilation(func: DefineFunOp, ctx: MLContext) -> list[str]:
    result_lst: list[str] = []
    if len(func.body.block.args) == 3:
        query_module = ModuleOp([])
        # getInstance x, y
        effect = ConstantBoolOp(False)
        arg_ops = getArgumentInstancesWithEffect(func, {})
        args = [arg.res for arg in arg_ops]

        forall_op = get_forall_self_annihilation_property(func, args[0], effect)

        # assert forall
        assert_forall_op = AssertOp(forall_op.res)

        all_ops = arg_ops + [forall_op, assert_forall_op] + [CheckSatOp()]

        query_module.body.block.add_ops(all_ops)
        FunctionCallInline(True, {}).apply(ctx, query_module)

        result, result_str = get_model(ctx, query_module)

        def print_self_annihilation_element(
            result: bool, func_name: str, s: str
        ) -> str:
            if result:
                int_val = get_int_val_from_smt_return(s)
                return "{func_name}(x, x) == {int_val}".format(
                    func_name=func_name, int_val=int_val
                )
            return "N/A"

        print(
            "Self Annihilation Element Check result: ",
            print_self_annihilation_element(result, func.fun_name.data, result_str),
        )
        if result:
            result_lst.append(get_int_val_from_smt_return(result_str))
    return result_lst


def check_all_property(
    func: DefineFunOp,
    func_name_to_func: dict[str, FuncOp],
    ctx: MLContext,
    properties: list[OperationProperties],
):
    comm = check_commutativity(func, ctx)
    func_name = func.fun_name.data
    cur_op = func_name_to_func[func_name].body.block.ops.first
    opp = OperationProperties(cur_op)
    opp.associativity = check_associativity(func, ctx)
    opp.involution = check_involution(func, ctx)
    opp.idempotence = check_idempotence(func, ctx)
    opp.absorbing_elements = check_absorbing_element(func, ctx, comm)
    opp.identity_elements = check_identity_element(func, ctx, comm)
    opp.self_annihilation_elements = check_self_annihilation(func, ctx)
    properties.append(opp)


def main() -> None:
    global ctx
    ctx = MLContext()
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    # Register all dialects
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(SMTDialect)
    ctx.load_dialect(SMTBitVectorDialect)
    ctx.load_dialect(SMTUtilsDialect)
    ctx.load_dialect(Transfer)
    ctx.load_dialect(Index)
    ctx.load_dialect(Comb)
    ctx.load_dialect(HW)

    # Parse the files
    module = parse_file(ctx, args.transfer_functions)
    assert isinstance(module, ModuleOp)

    func_name_to_func: dict[str, FuncOp] = {}
    for op in module.ops:
        if isinstance(op, FuncOp):
            func_name_to_func[op.sym_name.data] = op

    smt_module = module.clone()
    assert isinstance(smt_module, ModuleOp)
    lowerToSMTModule(smt_module, 8, ctx)
    operation_properties: list[OperationProperties] = []
    for op in smt_module.ops:
        if isinstance(op, DefineFunOp):
            print("Current check: ", op.fun_name)
            check_all_property(op, func_name_to_func, ctx, operation_properties)
            print("========")
    for opp in operation_properties:
        printer = PropertyPrinter(opp)
        print(printer.generate_interface())
