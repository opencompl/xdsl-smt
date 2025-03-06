import os
from os import path
from subprocess import run, PIPE
from enum import Enum, auto

from xdsl_smt.utils.compare_result import CompareResult


class AbstractDomain(Enum):
    KnownBits = auto()
    ConstantRange = auto()

    def __str__(self) -> str:
        return self.name


llvm_bin_dir: str = ""


def get_build_cmd() -> list[str]:
    has_libclang = (
        run(["ldconfig", "-p"], stdout=PIPE)
        .stdout.decode("utf-8")
        .find("libclang.so.19")
    )

    llvm_include_dir = (
        run(
            [llvm_bin_dir + "llvm-config", "--includedir"],
            stdout=PIPE,
        )
        .stdout.decode("utf-8")
        .split("\n")[0]
    )

    if llvm_bin_dir != "" or has_libclang == -1:
        all_llvm_link_flags = (
            run(
                [
                    llvm_bin_dir + "llvm-config",
                    "--ldflags",
                    "--libdir",
                    "--libs",
                    "--system-libs",
                ],
                stdout=PIPE,
            )
            .stdout.decode("utf-8")
            .split("\n")
        )
        all_llvm_link_flags = [x for x in all_llvm_link_flags if x != ""]
        lib_dir = all_llvm_link_flags[1]
        llvm_link_libs = all_llvm_link_flags[2].split(" ")

        llvm_link_flags = [all_llvm_link_flags[0]] + [
            x for x in llvm_link_libs if ("LLVMSupport" in x)
        ]

        build_cmd = [
            llvm_bin_dir + "clang++",
            "-std=c++20",
            "-O1",
            f"-I{llvm_include_dir}",
            f"-I{llvm_bin_dir}../include",
            "-L",
            f"{llvm_bin_dir}../lib",
            "../src/main.cpp",
            "-o",
            "EvalEngine",
            f"-Wl,-rpath,{lib_dir}",
        ] + llvm_link_flags
    else:
        llvm_link_flags = (
            run(
                ["llvm-config", "--ldflags", "--libs", "--system-libs"],
                stdout=PIPE,
            )
            .stdout.decode("utf-8")
            .split("\n")
        )
        llvm_link_flags = [x for x in llvm_link_flags if x != ""]
        build_cmd = [
            "clang++",
            "-std=c++20",
            "-O1",
            f"-I{llvm_include_dir}",
            "../src/main.cpp",
            "-o",
            "EvalEngine",
        ] + llvm_link_flags

    return build_cmd


def make_xfer_header(concrete_op: str) -> str:
    includes = """
    #include <llvm/ADT/APInt.h>
    #include <vector>
    #include "AbstVal.cpp"
    using llvm::APInt;
    """

    conc_op_wrapper = """
    unsigned int concrete_op_wrapper(const unsigned int a, const unsigned int b) {
      return concrete_op(APInt(32, a), APInt(32, b)).getZExtValue();
    }
    """

    return includes + concrete_op + conc_op_wrapper


def make_xfer_wrapper(func_names: list[str], wrapper_name: str) -> str:
    func_sig = (
        "std::vector<Domain> "
        + wrapper_name
        + "_wrapper(const Domain &lhs, const Domain &rhs)"
    )

    def make_func_call(x: str) -> str:
        return f"const std::vector<llvm::APInt> res_v_{x} = {x}" + "(lhs.v, rhs.v);"

    def make_res(x: str) -> str:
        return f"Domain res_{x}(res_v_{x});"

    func_calls = "\n".join([make_func_call(x) for x in func_names])
    results = "\n".join([make_res(x) for x in func_names])
    return_elems = ", ".join([f"res_{x}" for x in func_names])
    return_statment = "return {%s};" % return_elems

    return func_sig + "{" + f"\n{func_calls}\n{results}\n{return_statment}" + "}"


def eval_transfer_func(
    xfer_names: list[str],
    xfer_srcs: list[str],
    concrete_op_expr: str,
    ref_xfer_names: list[str],
    ref_xfer_srcs: list[str],
    domain: AbstractDomain,
    bitwidth: int,
    helper_funcs: list[str] | None = None,
) -> list[CompareResult]:
    func_to_eval_wrapper_name = "synth_function"
    ref_func_wrapper_name = "ref_function"
    ref_func_suffix = "REF"

    transfer_func_header = make_xfer_header(concrete_op_expr)
    transfer_func_header += f"\ntypedef {domain}<{bitwidth}> Domain;\n"
    transfer_func_header += f"\nunsigned int numFuncs = {len(xfer_names)};\n"

    # rename the transfer functions
    ref_xfer_srcs = [
        src.replace(nm, f"{nm}_{ref_func_suffix}_{i}")
        for i, (nm, src) in enumerate(zip(ref_xfer_names, ref_xfer_srcs))
    ]
    ref_xfer_names = [
        f"{nm}_{ref_func_suffix}_{i}" for i, nm in enumerate(ref_xfer_names)
    ]

    # create the wrapper
    ref_xfer_func_wrapper = make_xfer_wrapper(ref_xfer_names, ref_func_wrapper_name)

    # rename the transfer functions
    xfer_srcs = [
        src.replace(nm, f"{nm}_{i}")
        for i, (nm, src) in enumerate(zip(xfer_names, xfer_srcs))
    ]
    xfer_names = [f"{nm}_{i}" for i, nm in enumerate(xfer_names)]

    # create the wrapper
    xfer_func_wrapper = make_xfer_wrapper(xfer_names, func_to_eval_wrapper_name)

    all_xfer_src = "\n".join(xfer_srcs + ref_xfer_srcs)

    all_helper_funcs_src = ""
    if helper_funcs:
        all_helper_funcs_src = "\n".join(helper_funcs)

    base_dir = path.join("xdsl_smt", "eval_engine")
    cur_dir = os.getcwd()
    synth_code_path = path.join(cur_dir, base_dir, "src", "synth.cpp")

    with open(synth_code_path, "w") as f:
        f.write(
            f"{transfer_func_header}\n{all_helper_funcs_src}\n{all_xfer_src}\n{xfer_func_wrapper}\n{ref_xfer_func_wrapper}"
        )

    try:
        os.mkdir(path.join(cur_dir, base_dir, "build"))
    except FileExistsError:
        pass

    os.chdir(path.join(base_dir, "build"))

    run(get_build_cmd(), stdout=PIPE)
    eval_output = run(["./EvalEngine"], stdout=PIPE, stderr=PIPE)

    if eval_output.returncode != 0:
        print("EvalEngine failed with this error:")
        print(eval_output.stderr.decode("utf-8"), end="")
        exit(eval_output.returncode)

    def get_floats(s: str) -> list[int]:
        return eval(s)

    os.chdir(cur_dir)

    eval_output_lines = eval_output.stdout.decode("utf-8").split("\n")
    sounds = get_floats(eval_output_lines[1])
    precs = get_floats(eval_output_lines[3])
    exact = get_floats(eval_output_lines[5])
    num_cases = get_floats(eval_output_lines[7])
    unsolved_sounds = get_floats(eval_output_lines[9])
    unsolved_precs = get_floats(eval_output_lines[11])
    unsolved_exact = get_floats(eval_output_lines[13])
    unsolved_num_cases = get_floats(eval_output_lines[15])

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
            bitwidth,
        )
        for i in range(len(sounds))
    ]

    return cmp_results


def main():
    constraint_func = """
    bool op_constraint(APInt _arg0, APInt _arg1){
        return true;
    }
    """

    concrete_op = """
    APInt concrete_op(APInt a, APInt b) {
        return a+b;
    }
    """

    transfer_func_name = "cr_add"
    transfer_func_src = """
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
    """

    names = [transfer_func_name]
    srcs = [transfer_func_src]
    ref_names: list[str] = []  # TODO
    ref_srcs: list[str] = []  # TODO
    results = eval_transfer_func(
        names,
        srcs,
        f"{concrete_op}\n{constraint_func}",
        ref_names,
        ref_srcs,
        AbstractDomain.ConstantRange,
        4,
    )

    for res in results:
        print(res)
        print(f"cost:                  {res.get_cost():.04f}")
        print(f"sound prop:            {res.get_sound_prop():.04f}")
        print(f"exact prop:            {res.get_exact_prop():.04f}")
        print(f"edit dis avg:          {res.get_edit_dis_avg():.04f}")
        print(f"unsolved exact prop:   {res.get_unsolved_exact_prop():.04f}")
        print(f"unsolved sound prop:   {res.get_unsolved_sound_prop():.04f}")
        print(f"unsolved edit dis avg: {res.get_unsolved_edit_dis_avg():.04f}")


if __name__ == "__main__":
    main()
