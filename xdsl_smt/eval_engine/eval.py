import os
from os import path
from subprocess import run, PIPE

from xdsl_smt.utils.compare_result import CompareResult


def get_build_cmd() -> list[str]:
    has_libclang = (
        run(["ldconfig", "-p"], stdout=PIPE)
        .stdout.decode("utf-8")
        .find("libclang.so.19")
    )

    llvm_include_dir = (
        run(
            ["llvm-config", "--includedir"],
            stdout=PIPE,
        )
        .stdout.decode("utf-8")
        .split("\n")[0]
    )

    if has_libclang == -1:
        all_llvm_link_flags = (
            run(
                ["llvm-config", "--ldflags", "--libdir", "--libs", "--system-libs"],
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
            "clang++",
            "-std=c++23",
            f"-I{llvm_include_dir}",
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
            "-std=c++23",
            f"-I{llvm_include_dir}",
            "../src/main.cpp",
            "-o",
            "EvalEngine",
        ] + llvm_link_flags

    return build_cmd


def make_xfer_header(concrete_op: str, num_funcs: int) -> str:
    includes = """
    #include <llvm/ADT/APInt.h>
    #include <tuple>
    #include <vector>
    #include "AbstVal.cpp"
    using llvm::APInt;
    """

    conc_op_wrapper = """
    uint8_t concrete_op_wrapper(const uint8_t a, const uint8_t b) {
      return concrete_op(APInt(8, a), APInt(8, b)).getZExtValue();
    }
    """

    num_funcs_const = f"unsigned int numFuncs = {num_funcs};"

    return includes + concrete_op + conc_op_wrapper + num_funcs_const


def make_xfer_wrapper(func_names: list[str], wrapper_name: str) -> str:
    func_sig = (
        "std::vector<AbstVal> "
        + wrapper_name
        + "_wrapper(const AbstVal &lhs, const AbstVal &rhs)"
    )

    def make_func_call(x: str) -> str:
        return f"const std::vector<llvm::APInt> res_v_{x} = {x}" + "(lhs.v, rhs.v);"

    def make_res(x: str) -> str:
        return f"AbstVal res_{x}(lhs.domain, res_v_{x});"

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
) -> list[CompareResult]:
    func_to_eval_wrapper_name = "synth_function"
    ref_func_wrapper_name = "ref_function"

    transfer_func_header = make_xfer_header(concrete_op_expr, len(xfer_names))

    # rename the transfer functions
    ref_xfer_srcs = [
        src.replace(nm, f"{nm}_{i}")
        for i, (nm, src) in enumerate(zip(ref_xfer_names, ref_xfer_srcs))
    ]
    ref_xfer_names = [f"{nm}_{i}" for i, nm in enumerate(ref_xfer_names)]

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

    base_dir = path.join("xdsl_smt", "eval_engine")
    cur_dir = os.getcwd()
    synth_code_path = path.join(cur_dir, base_dir, "src", "synth.cpp")

    with open(synth_code_path, "w") as f:
        f.write(
            f"{transfer_func_header}\n{all_xfer_src}\n{xfer_func_wrapper}\n{ref_xfer_func_wrapper}"
        )

    try:
        os.mkdir(path.join(cur_dir, base_dir, "build"))
    except FileExistsError:
        pass

    os.chdir(path.join(base_dir, "build"))

    run(get_build_cmd(), stdout=PIPE)
    eval_output = run(
        # TODO pass domain as a flag to this function
        # also make domain an enum
        ["./EvalEngine", "--domain", "ConstantRange"],
        stdout=PIPE,
        stderr=PIPE,
    )

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
        )
        for i in range(len(sounds))
    ]

    # return sounds, precs, exact, num_cases, unsolved_sounds, unsolved_precs, unsolved_exact, unsolved_num_cases
    return cmp_results


def main():
    concrete_op = """
    APInt concrete_op(APInt a, APInt b) {
        return a+b;
    }
    """

    transfer_func_name = "llm_wrapper"
    transfer_func_src = """
        #include <llvm/ADT/APInt.h>
        #include <tuple>
        #include <vector>

        using llvm::APInt;

        std::tuple<int, int> abstract_udiv(std::tuple<int, int> &lhs, std::tuple<int, int> &rhs) {
          return std::make_tuple(0, 0);
        }

        std::vector<APInt> llm_wrapper(std::vector<APInt> arg0,
                                       std::vector<APInt> arg1) {
          auto lhs = std::tuple(static_cast<int>(arg0[0].getZExtValue()),
                                static_cast<int>(arg0[1].getZExtValue()));
          auto rhs = std::tuple(static_cast<int>(arg1[0].getZExtValue()),
                                static_cast<int>(arg1[1].getZExtValue()));
          auto res = abstract_udiv(lhs, rhs);

          APInt res_0 = APInt(4, static_cast<uint64_t>(std::get<0>(res)));
          APInt res_1 = APInt(4, static_cast<uint64_t>(std::get<1>(res)));

          return {res_0, res_1};
        }
    """

    names = [transfer_func_name]
    srcs = [transfer_func_src]
    ref_names: list[str] = []  # TODO
    ref_srcs: list[str] = []  # TODO
    a = eval_transfer_func(names, srcs, concrete_op, ref_names, ref_srcs)

    for res in a:
        print(f"cost:                  {res.get_cost():.04f}")
        print(f"sound prop:            {res.get_sound_prop():.04f}")
        print(f"exact prop:            {res.get_exact_prop():.04f}")
        print(f"edit dis avg:          {res.get_edit_dis_avg():.04f}")
        print(f"unsolved exact prop:   {res.get_unsolved_exact_prop():.04f}")
        print(f"unsolved sound prop:   {res.get_unsolved_sound_prop():.04f}")
        print(f"unsolved edit dis avg: {res.get_unsolved_edit_dis_avg():.04f}")


if __name__ == "__main__":
    main()
