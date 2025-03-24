# this is a temporary place to have unit tests for the eval engine
# these should be turned into llvm-lit tests and moved to the tests dir
# run these tests via `python -m xdsl_smt.eval_engine.tester`
# TODO test at different bitwidths

from typing import NamedTuple
from xdsl_smt.eval_engine.eval import eval_transfer_func, AbstractDomain


class TestInput(NamedTuple):
    concrete_op: str
    domain: AbstractDomain
    functions: list[tuple[str, str]]
    expected_outputs: list[str]


cnc_or = 'extern "C" APInt concrete_op(APInt a, APInt b) { return a|b; }'
cnc_add = 'extern "C" APInt concrete_op(APInt a, APInt b) { return a+b; }'
cnc_sub = 'extern "C" APInt concrete_op(APInt a, APInt b) { return a-b; }'
cnc_xor = 'extern "C" APInt concrete_op(APInt a, APInt b) { return a^b; }'
cnc_and = 'extern "C" APInt concrete_op(APInt a, APInt b) { return a&b; }'
cnc_udiv = 'extern "C" APInt concrete_op(APInt a, APInt b) {return a.udiv(b);}'
cnc_urem = 'extern "C" APInt concrete_op(APInt a, APInt b) {return a.urem(b);}'
cnc_umin = 'extern "C" APInt concrete_op(APInt a,APInt b) {return APIntOps::umin(a,b);}'
cnc_umax = 'extern "C" APInt concrete_op(APInt a,APInt b) {return APIntOps::umax(a,b);}'

kb_and = (
    "kb_and",
    """
extern "C" Vec<2> kb_and(const Vec<2> arg0, const Vec<2> arg1) {
  APInt res_0 = arg0[0] | arg1[0];
  APInt res_1 = arg0[1] & arg1[1];
  return {res_0, res_1};
}
""",
)

kb_or = (
    "kb_or",
    """
extern "C" Vec<2> kb_or(const Vec<2> arg0, const Vec<2> arg1) {
  APInt res_0 = arg0[0] & arg1[0];
  APInt res_1 = arg0[1] | arg1[1];
  return {res_0, res_1};
}
""",
)

kb_xor = (
    "kb_xor",
    """
extern "C" Vec<2> kb_xor(const Vec<2> arg0, const Vec<2> arg1) {
  APInt res_0 = (arg0[0] & arg1[0]) | (arg0[1] & arg1[1]);
  APInt res_1 = (arg0[0] & arg1[1]) | (arg0[1] & arg1[0]);
  return {res_0, res_1};
}
""",
)

cr_add = (
    "cr_add",
    """
extern "C" Vec<2> cr_add(const Vec<2> arg0, const Vec<2> arg1) {
  bool res0_ov;
  bool res1_ov;
  APInt res0 = arg0[0].uadd_ov(arg1[0], res0_ov);
  APInt res1 = arg0[1].uadd_ov(arg1[1], res1_ov);
  if (res0.ugt(res1) || (res0_ov ^ res1_ov))
    return {APInt::getMinValue(arg0[0].getBitWidth()),
            APInt::getMaxValue(arg0[0].getBitWidth())};
  return {res0, res1};
}
""",
)

cr_sub = (
    "cr_sub",
    """
extern "C" Vec<2> cr_sub(const Vec<2> arg0, const Vec<2> arg1) {
  bool res0_ov;
  bool res1_ov;
  APInt res0 = arg0[0].usub_ov(arg1[1], res0_ov);
  APInt res1 = arg0[1].usub_ov(arg1[0], res1_ov);
  if (res0.ugt(res1) || (res0_ov ^ res1_ov))
    return {APInt::getMinValue(arg0[0].getBitWidth()),
            APInt::getMaxValue(arg0[0].getBitWidth())};
  return {res0, res1};
}
""",
)


def test(input: TestInput) -> None:
    names, srcs = zip(*input.functions)
    results = eval_transfer_func(
        list(names), list(srcs), [], [], [input.concrete_op], input.domain, 4
    )

    for n, r, e in zip(names, results, input.expected_outputs):
        if str(r) != e:
            print("Unit test failure:\n")
            print(f"Abstract domain: {input.domain}")
            print(f"Concrete function:\n{input.concrete_op}")
            print(f"Failed function source name: {n}")
            print(f"Expected: {e}")
            print(f"Got:      {r}")
            print("===================================================================")


kb_or_test = TestInput(
    cnc_or,
    AbstractDomain.KnownBits,
    [kb_xor, kb_and, kb_or],
    [
        "all: 6561	s: 4096	e: 1296	p: 11664	unsolved:6480	us: 4015	ue: 1215	up: 11664	basep: 17496",
        "all: 6561	s: 625	e: 81	p: 23328	unsolved:6480	us: 624	ue: 80	up: 23112	basep: 17496",
        "all: 6561	s: 6561	e: 6561	p: 0	unsolved:6480	us: 6480	ue: 6480	up: 0	basep: 17496",
    ],
)

kb_and_test = TestInput(
    cnc_and,
    AbstractDomain.KnownBits,
    [kb_xor, kb_and, kb_or],
    [
        "all: 6561	s: 1296	e: 256	p: 23328	unsolved:6480	us: 1215	ue: 175	up: 23328	basep: 17496",
        "all: 6561	s: 6561	e: 6561	p: 0	unsolved:6480	us: 6480	ue: 6480	up: 0	basep: 17496",
        "all: 6561	s: 625	e: 81	p: 23328	unsolved:6480	us: 624	ue: 80	up: 23112	basep: 17496",
    ],
)

kb_xor_test = TestInput(
    cnc_xor,
    AbstractDomain.KnownBits,
    [kb_xor, kb_and, kb_or],
    [
        "all: 6561	s: 6561	e: 6561	p: 0	unsolved:5936	us: 5936	ue: 5936	up: 0	basep: 11664",
        "all: 6561	s: 256	e: 256	p: 23328	unsolved:5936	us: 175	ue: 175	up: 22328	basep: 11664",
        "all: 6561	s: 1296	e: 1296	p: 11664	unsolved:5936	us: 1215	ue: 1215	up: 10664	basep: 11664",
    ],
)

kb_add_test = TestInput(
    cnc_add,
    AbstractDomain.KnownBits,
    [kb_xor, kb_and, kb_or],
    [
        "all: 6561	s: 2625	e: 2625	p: 6620	unsolved:4220	us: 2000	ue: 2000	up: 4328	basep: 7692",
        "all: 6561	s: 121	e: 121	p: 20018	unsolved:4220	us: 40	ue: 40	up: 15358	basep: 7692",
        "all: 6561	s: 897	e: 897	p: 14974	unsolved:4220	us: 816	ue: 816	up: 9554	basep: 7692",
    ],
)

cr_add_test = TestInput(
    cnc_add,
    AbstractDomain.ConstantRange,
    [cr_add, cr_sub],
    [
        "all: 18769	s: 18769	e: 18769	p: 0	unsolved:6920	us: 6920	ue: 6920	up: 0	basep: 20864",
        "all: 18769	s: 12224	e: 9179	p: 30596	unsolved:6920	us: 3420	ue: 375	up: 22212	basep: 20864",
    ],
)

cr_sub_test = TestInput(
    cnc_sub,
    AbstractDomain.ConstantRange,
    [cr_sub, cr_add],
    [
        "all: 18769	s: 18769	e: 18769	p: 0	unsolved:6920	us: 6920	ue: 6920	up: 0	basep: 20864",
        "all: 18769	s: 12224	e: 9179	p: 30596	unsolved:6920	us: 3420	ue: 375	up: 22212	basep: 20864",
    ],
)


if __name__ == "__main__":
    test(kb_or_test)
    test(kb_and_test)
    test(kb_xor_test)
    test(kb_add_test)
    test(cr_add_test)
    test(cr_sub_test)
