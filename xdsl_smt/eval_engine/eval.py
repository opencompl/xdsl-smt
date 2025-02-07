import os
from os import path
from subprocess import run, PIPE


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


def make_xfer_header(concrete_op: str) -> str:
    includes = """
    #include <llvm/ADT/APInt.h>
    #include <tuple>
    #include <vector>
    using llvm::APInt;
    """
    conc_op_wrapper = """
    uint8_t concrete_op_wrapper(const uint8_t a, const uint8_t b) {
      return concrete_op(APInt(8, a), APInt(8, b)).getZExtValue();
    }
    """
    return includes + concrete_op + conc_op_wrapper


def make_xfer_wrapper(func_names: list[str]) -> str:
    func_sig = "std::vector<AbstVal> synth_function_wrapper(const AbstVal &lhs, const AbstVal &rhs)"

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
) -> tuple[list[int], list[int], list[int], list[int]]:
    transfer_func_header = make_xfer_header(concrete_op_expr)

    xfer_srcs = [
        src.replace(nm, f"{nm}_{i}")
        for i, (nm, src) in enumerate(zip(xfer_names, xfer_srcs))
    ]
    xfer_names = [f"{nm}_{i}" for i, nm in enumerate(xfer_names)]
    xfer_func_wrapper = make_xfer_wrapper(xfer_names)
    all_xfer_src = "\n".join(xfer_srcs)

    base_dir = path.join("xdsl_smt", "eval_engine")
    cur_dir = os.getcwd()
    synth_code_path = path.join(cur_dir, base_dir, "src", "synth.cpp")

    with open(synth_code_path, "w") as f:
        f.write(f"{transfer_func_header}\n{all_xfer_src}\n{xfer_func_wrapper}")

    try:
        os.mkdir(path.join(cur_dir, base_dir, "build"))
    except FileExistsError:
        pass

    os.chdir(path.join(base_dir, "build"))

    run(get_build_cmd(), stdout=PIPE)
    eval_output = run(["./EvalEngine"], stdout=PIPE)

    def get_floats(s: str) -> list[int]:
        return eval(s)

    os.chdir(cur_dir)

    eval_output_lines = eval_output.stdout.decode("utf-8").split("\n")
    sounds = get_floats(eval_output_lines[1])
    precs = get_floats(eval_output_lines[3])
    exact = get_floats(eval_output_lines[5])
    num_cases = get_floats(eval_output_lines[7])

    return sounds, precs, exact, num_cases


# cost function on `synth_transfer` branch as of feb 05
# git commit 9df592f15b23e1759365d20790760ff192d232d7
def compute_cost(soundness: float, precision: float) -> float:
    a: float = 1
    b: float = 4
    return (a * (1 - soundness) + b * (1 - precision)) / (a + b)


'''
if __name__ == "__main__":
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
    num_unsound, imprecision, num_exact, num_cases = eval_transfer_func(
        names, srcs, concrete_op
    )

    soundness_percent = 1 - (num_unsound[0] / num_cases[0])
    precision_percent = num_exact[0] / num_cases[0]

    proposed_cost = compute_cost(soundness_percent, precision_percent)

    print(f"num_unsound:   {num_unsound[0]}")
    print(f"imprecision:   {imprecision[0]}")
    print(f"num_exact:     {num_exact[0]}")
    print(f"num_cases:     {num_cases[0]}")
    print(f"proposed_cost: {proposed_cost:.04f}")
'''
