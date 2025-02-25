# this is a temporary place to have unit tests for the eval engine
# these should be turned into llvm-lit tests and moved to the tests dir
# run these tests via `python -m xdsl_smt.eval_engine.tester`

from typing import NamedTuple
from xdsl_smt.eval_engine.eval import eval_transfer_func, AbstractDomain

# TODO test for known_bits and const_range and int_mod
# TODO test at different bitwidths
# TODO add more abst domain srcs to test with


class TestInput(NamedTuple):
    concrete_op: str
    domain: AbstractDomain
    functions: list[tuple[str, str]]
    expected_outputs: list[str]


# maybe just make a dict
concrete_or = "APInt concrete_op(APInt a, APInt b) { return a|b; }"
concrete_add = "APInt concrete_op(APInt a, APInt b) { return a+b; }"
concrete_sub = "APInt concrete_op(APInt a, APInt b) { return a-b; }"
concrete_xor = "APInt concrete_op(APInt a, APInt b) { return a^b; }"
concrete_and = "APInt concrete_op(APInt a, APInt b) { return a&b; }"
concrete_udiv = "APInt concrete_op(APInt a, APInt b) { return a.udiv(b); }"
concrete_urem = "APInt concrete_op(APInt a, APInt b) { return a.urem(b); }"
concrete_umin = "APInt concrete_op(APInt a,APInt b) {return llvm::APIntOps::umin(a,b);}"
concrete_umax = "APInt concrete_op(APInt a,APInt b) {return llvm::APIntOps::umax(a,b);}"

kb_and = (
    "kb_and",
    """
std::vector<APInt> kb_and(std::vector<APInt> arg0, std::vector<APInt> arg1) {
  APInt res_0 = arg0[0] | arg1[0];
  APInt res_1 = arg0[1] & arg1[1];
  return {res_0, res_1};
}
""",
)

kb_or = (
    "kb_or",
    """
std::vector<APInt> kb_or(std::vector<APInt> arg0, std::vector<APInt> arg1) {
  APInt res_0 = arg0[0] & arg1[0];
  APInt res_1 = arg0[1] | arg1[1];
  return {res_0, res_1};
}
""",
)

kb_xor = (
    "kb_xor",
    """
std::vector<APInt> kb_xor(std::vector<APInt> arg0, std::vector<APInt> arg1) {
  APInt res_0 = (arg0[0] & arg1[0]) | (arg0[1] & arg1[1]);
  APInt res_1 = (arg0[0] & arg1[1]) | (arg0[1] & arg1[0]);
  return {res_0, res_1};
}
""",
)

cr_add = (
    "cr_add",
    """
std::vector<APInt> cr_add(std::vector<APInt> arg0, std::vector<APInt> arg1) {
  bool res0_ov;
  bool res1_ov;
  APInt res0 = arg0[0].uadd_ov(arg1[0], res0_ov);
  APInt res1 = arg0[1].uadd_ov(arg1[1], res1_ov);
  if (res0.ugt(res1) || (res0_ov ^ res1_ov))
    return {llvm::APInt::getMinValue(arg0[0].getBitWidth()),
            llvm::APInt::getMaxValue(arg0[0].getBitWidth())};
  return {res0, res1};
}
""",
)

cr_sub = (
    "cr_sub",
    """
std::vector<APInt> cr_sub(std::vector<APInt> arg0, std::vector<APInt> arg1) {
  bool res0_ov;
  bool res1_ov;
  APInt res0 = arg0[0].usub_ov(arg1[1], res0_ov);
  APInt res1 = arg0[1].usub_ov(arg1[0], res1_ov);
  if (res0.ugt(res1) || (res0_ov ^ res1_ov))
    return {llvm::APInt::getMinValue(arg0[0].getBitWidth()),
            llvm::APInt::getMaxValue(arg0[0].getBitWidth())};
  return {res0, res1};
}
""",
)


def test(input: TestInput) -> None:
    constraint_func = """
    bool op_constraint(APInt _arg0, APInt _arg1){
        return true;
    }
    """

    names, srcs = zip(*input.functions)
    results = eval_transfer_func(
        list(names),
        list(srcs),
        f"{input.concrete_op}\n{constraint_func}",
        [],
        [],
        input.domain,
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
    concrete_or,
    AbstractDomain.KnownBits,
    [kb_xor, kb_and, kb_or],
    [
        "all: 6561	s: 4096	e: 1296	p: 11664	unsolved:6480	us: 4015	ue: 1215	up: 11664",
        "all: 6561	s: 625	e: 81	p: 23328	unsolved:6480	us: 624	ue: 80	up: 23112",
        "all: 6561	s: 6561	e: 6561	p: 0	unsolved:6480	us: 6480	ue: 6480	up: 0",
    ],
)

kb_and_test = TestInput(
    concrete_and,
    AbstractDomain.KnownBits,
    [kb_xor, kb_and, kb_or],
    [
        "all: 6561	s: 1296	e: 256	p: 23328	unsolved:6480	us: 1215	ue: 175	up: 23328",
        "all: 6561	s: 6561	e: 6561	p: 0	unsolved:6480	us: 6480	ue: 6480	up: 0",
        "all: 6561	s: 625	e: 81	p: 23328	unsolved:6480	us: 624	ue: 80	up: 23112",
    ],
)

kb_xor_test = TestInput(
    concrete_xor,
    AbstractDomain.KnownBits,
    [kb_xor, kb_and, kb_or],
    [
        "all: 6561	s: 6561	e: 6561	p: 0	unsolved:5936	us: 5936	ue: 5936	up: 0",
        "all: 6561	s: 256	e: 256	p: 23328	unsolved:5936	us: 175	ue: 175	up: 22328",
        "all: 6561	s: 1296	e: 1296	p: 11664	unsolved:5936	us: 1215	ue: 1215	up: 10664",
    ],
)

kb_add_test = TestInput(
    concrete_add,
    AbstractDomain.KnownBits,
    [kb_xor, kb_and, kb_or],
    [
        "all: 6561	s: 2625	e: 2625	p: 6620	unsolved:4220	us: 2000	ue: 2000	up: 4328",
        "all: 6561	s: 121	e: 121	p: 20018	unsolved:4220	us: 40	ue: 40	up: 15358",
        "all: 6561	s: 897	e: 897	p: 14974	unsolved:4220	us: 816	ue: 816	up: 9554",
    ],
)

cr_add_test = TestInput(
    concrete_add,
    AbstractDomain.ConstantRange,
    [cr_add, cr_sub],
    [
        "all: 18769	s: 18769	e: 18769	p: 0	unsolved:6920	us: 6920	ue: 6920	up: 0",
        "all: 18769	s: 12224	e: 9179	p: 30596	unsolved:6920	us: 3420	ue: 375	up: 22212",
    ],
)

cr_sub_test = TestInput(
    concrete_sub,
    AbstractDomain.ConstantRange,
    [cr_sub, cr_add],
    [
        "all: 18769	s: 18769	e: 18769	p: 0	unsolved:6920	us: 6920	ue: 6920	up: 0",
        "all: 18769	s: 12224	e: 9179	p: 30596	unsolved:6920	us: 3420	ue: 375	up: 22212",
    ],
)

if __name__ == "__main__":
    test(kb_or_test)
    test(kb_and_test)
    test(kb_xor_test)
    test(kb_add_test)
    test(cr_add_test)
    test(cr_sub_test)
