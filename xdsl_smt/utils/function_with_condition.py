from dataclasses import dataclass, field
from typing import Callable
from xdsl.dialects.builtin import StringAttr, i1
from xdsl.dialects.func import FuncOp, CallOp, ReturnOp
from xdsl_smt.dialects.transfer import AbstractValueType, GetOp, SelectOp, MakeOp


@dataclass
class FunctionWithCondition:
    """
    Class for a transfer function f in the form of "f(a) := if (cond) then func(a) else Top"
    """

    func: FuncOp
    cond: FuncOp | None = None

    """
    This func_name is used in generation the whole function.
    """
    func_name: str = field(init=False)

    def set_func_name(self, new_func_name: str):
        self.func_name = new_func_name
        self.func.sym_name = StringAttr(new_func_name + "_body")
        if self.cond is not None:
            self.cond.sym_name = StringAttr(new_func_name + "_cond")

    def to_str(self, eliminate_dead_code: Callable[[FuncOp], FuncOp]):
        cond_str = (
            "True\n" if self.cond is None else str(eliminate_dead_code(self.cond))
        )
        return f"Cond:\n{cond_str}\nFunc:{str(eliminate_dead_code(self.func))}"

    def get_function(self) -> FuncOp:
        """
        Because select operation only works on TransferIntegertype, so we have to decouple all result obtained from getTop
        and call_body
        The whole function first get top and get all its elements
        Next, it calls body and get all its element
        Finally, it selects element by the condition and returns it

        TODO: Add select support on AbstractValues, so this function can be simplified.
        """

        whole_function = FuncOp(self.func_name, self.func.function_type)
        args = whole_function.args

        if self.cond is None:
            call_op = CallOp(
                self.func.sym_name.data, args, self.func.function_type.outputs.data
            )
            assert len(call_op.results) == 1
            call_res = call_op.results[0]
            return_op = ReturnOp(call_res)
            whole_function.body.block.add_ops([call_op, return_op])
            return whole_function

        call_top_op = CallOp("getTop", [args[0]], self.func.function_type.outputs.data)
        assert len(call_top_op.results) == 1
        top_res = call_top_op.results[0]
        top_res_type = top_res.type

        top_res_element_ops: list[GetOp] = []
        assert isinstance(top_res_type, AbstractValueType)
        for i in range(top_res_type.get_num_fields()):
            top_res_element_ops.append(GetOp(top_res, i))

        call_body_op = CallOp(
            self.func.sym_name.data, args, self.func.function_type.outputs.data
        )
        assert len(call_body_op.results) == 1
        body_res = call_body_op.results[0]
        body_res_type = body_res.type

        body_res_element_ops: list[GetOp] = []
        assert body_res_type == top_res_type
        for i in range(top_res_type.get_num_fields()):
            body_res_element_ops.append(GetOp(body_res, i))

        res_element_ops: list[SelectOp] = []
        call_cond_op = CallOp(
            self.cond.sym_name.data, args, self.cond.function_type.outputs.data
        )
        assert len(call_cond_op.results) == 1
        cond_res = call_cond_op.results[0]

        assert cond_res.type == i1
        for top_ele_op, body_ele_op in zip(top_res_element_ops, body_res_element_ops):
            res_element_ops.append(
                SelectOp(cond_res, body_ele_op.result, top_ele_op.result)
            )

        make_op = MakeOp([res_ele.result for res_ele in res_element_ops])
        return_op = ReturnOp(make_op)
        whole_function.body.block.add_ops(
            [call_top_op]
            + top_res_element_ops
            + [call_body_op]
            + body_res_element_ops
            + [call_cond_op]
            + res_element_ops
            + [make_op, return_op]
        )
        return whole_function

    def get_function_str(
        self,
        lower_to_cpp: Callable[[FuncOp], str],
        eliminate_dead_code: Callable[[FuncOp], FuncOp],
    ) -> tuple[str, list[str]]:
        whole_function = self.get_function()
        whole_function_str = lower_to_cpp(eliminate_dead_code(whole_function))
        func_str = lower_to_cpp(eliminate_dead_code(self.func))
        if self.cond is None:
            return whole_function_str, [func_str]
        cond_str = lower_to_cpp(eliminate_dead_code(self.cond))
        return whole_function_str, [func_str, cond_str]
