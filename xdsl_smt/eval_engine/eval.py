from os import path
from subprocess import run, PIPE
from enum import Enum, auto
import re

from xdsl_smt.utils.compare_result import CompareResult


class AbstractDomain(Enum):
    KnownBits = auto()
    ConstantRange = auto()

    def __str__(self) -> str:
        return self.name


# silly little formatting things
def band_aid(code: str) -> str:
    id_p = r"([a-zA-Z_][a-zA-Z0-9_]*)"

    n_pre = 'extern "C" struct Ret '
    n_post = "(const APInt *const arg0, const APInt *const arg1)"

    o_pre = re.escape("std::vector<APInt> ")
    o_post = re.escape("(std::vector<APInt> arg0,std::vector<APInt> arg1)")
    p = rf"{o_pre}{id_p}{o_post}"
    code = re.sub(p, rf"{n_pre}\1{n_post}", code)

    o_post = re.escape("(std::vector<APInt> autogen0,std::vector<APInt> autogen1)")
    n_post = "(const APInt *const autogen0, const APInt *const autogen1)"
    p = rf"{o_pre}{id_p}{o_post}"
    code = re.sub(p, rf"{n_pre}\1{n_post}", code)

    middle = re.escape("=std::vector<APInt>{")
    sep = re.escape(",")
    end = re.escape("};")
    ret = re.escape("return ")
    semi = re.escape(";")

    p = rf"{o_pre}{id_p}{middle}{id_p}{sep}{id_p}{end}\s*{ret}{id_p}{semi}"
    code = re.sub(p, r"return {\2,\3};", code)

    old = "int getInstanceConstraint(std::vector<APInt> arg0,APInt inst)"
    new = "int getInstanceConstraint(const APInt *const arg0,APInt inst)"
    code = code.replace(old, new)

    old = "int getConstraint(std::vector<APInt> arg0)"
    new = "int getConstraint(const APInt *const arg0)"
    code = code.replace(old, new)

    old = "APInt concrete_op"
    new = 'extern "C" APInt concrete_op'
    code = code.replace(old, new)

    old = "std::vector<APInt> solution(std::vector<APInt> autogen0,std::vector<APInt> autogen1)"
    new = 'extern "C" struct Ret solution(const APInt *const  autogen0,const APInt *const  autogen1)'
    code = code.replace(old, new)

    p = rf"{o_pre}{id_p}{re.escape('=')}"
    code = re.sub(p, r"struct Ret \1=", code)

    return code


def eval_transfer_func(
    xfer_names: list[str],
    xfer_srcs: list[str],
    base_names: list[str],
    base_srcs: list[str],
    helper_srcs: list[str],
    domain: AbstractDomain,
    bitwidth: int,
) -> list[CompareResult]:
    base_dir = path.join("xdsl_smt", "eval_engine")
    engine_path = path.join(base_dir, "build", "eval_engine")
    if not path.exists(engine_path):
        raise FileExistsError(f"Eval Enigne not found at: {engine_path}")

    # rename the transfer functions
    base_srcs = [
        src.replace(nm, f"{nm}_base_{i}")
        for i, (nm, src) in enumerate(zip(base_names, base_srcs))
    ]
    base_names = [f"{nm}_base_{i}" for i, nm in enumerate(base_names)]

    # rename the transfer functions
    xfer_srcs = [
        src.replace(nm, f"{nm}_{i}")
        for i, (nm, src) in enumerate(zip(xfer_names, xfer_srcs))
    ]
    xfer_names = [f"{nm}_{i}" for i, nm in enumerate(xfer_names)]

    engine_params = ""
    engine_params += f"{domain}\n"
    engine_params += f"{bitwidth}\n"
    engine_params += f"{' '.join(xfer_names)}\n"
    engine_params += f"{' '.join(base_names)}\n"
    engine_params += "using A::APInt;\n"

    all_src = "\n".join(helper_srcs + xfer_srcs + base_srcs)
    engine_params += band_aid(all_src)

    eval_output = run(
        [engine_path],
        input=engine_params,
        text=True,
        stdout=PIPE,
        stderr=PIPE,
    )

    if eval_output.returncode != 0:
        print("EvalEngine failed with this error:")
        print(eval_output.stderr, end="")
        exit(eval_output.returncode)

    def get_floats(s: str) -> list[int]:
        return eval(s)

    eval_output_lines = eval_output.stdout.split("\n")
    sounds = get_floats(eval_output_lines[1])
    precs = get_floats(eval_output_lines[3])
    exact = get_floats(eval_output_lines[5])
    num_cases = get_floats(eval_output_lines[7])
    unsolved_sounds = get_floats(eval_output_lines[9])
    unsolved_precs = get_floats(eval_output_lines[11])
    unsolved_exact = get_floats(eval_output_lines[13])
    unsolved_num_cases = get_floats(eval_output_lines[15])
    base_precs = get_floats(eval_output_lines[17])

    assert len(sounds) > 0, f"No output from EvalEngine: {eval_output}"
    assert (
        len(sounds)
        == len(precs)
        == len(exact)
        == len(num_cases)
        == len(unsolved_sounds)
        == len(unsolved_precs)
        == len(unsolved_exact)
        == len(unsolved_num_cases)
        == len(base_precs)
    ), f"EvalEngine output mismatch: {eval_output}"

    cmp_results: list[CompareResult] = [
        CompareResult(
            num_cases[i],
            sounds[i],
            exact[i],
            precs[i],
            unsolved_num_cases[i],
            unsolved_sounds[i],
            unsolved_exact[i],
            unsolved_precs[i],
            base_precs[i],
            bitwidth,
        )
        for i in range(len(sounds))
    ]

    return cmp_results
