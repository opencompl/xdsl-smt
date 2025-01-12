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
    #include <llvm/Support/KnownBits.h>
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
    func_sig = "std::vector<llvm::KnownBits> synth_function_wrapper(const llvm::KnownBits &lhs, const llvm::KnownBits &rhs)"

    def make_func_call(x: str) -> str:
        return (
            f"const std::vector<llvm::APInt> res_v_{x} = {x}"
            + "({lhs.Zero, lhs.One}, {rhs.Zero, rhs.One});"
        )

    def make_res(x: str) -> str:
        return f"llvm::KnownBits res_{x};\nres_{x}.Zero = res_v_{x}[0];\nres_{x}.One = res_v_{x}[1];\n"

    func_calls = "\n".join([make_func_call(x) for x in func_names])
    results = "\n".join([make_res(x) for x in func_names])
    return_elems = ", ".join([f"res_{x}" for x in func_names])
    return_statment = "return {%s};" % return_elems

    return func_sig + "{" + f"\n{func_calls}\n{results}\n{return_statment}" + "}"


def eval_transfer_func(
    xfer_names: list[str],
    xfer_srcs: list[str],
    concrete_op_expr: str,
) -> tuple[list[float], list[float]]:
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

    def get_floats(s: str) -> list[float]:
        return eval(s)

    os.chdir(cur_dir)

    eval_output_lines = eval_output.stdout.decode("utf-8").split("\n")
    sounds = get_floats(eval_output_lines[1])
    precs = get_floats(eval_output_lines[3])

    return sounds, precs


'''
if __name__ == "__main__":
    concrete_op = """
    APInt concrete_op(APInt a, APInt b) {
        return a+b;
    }
    """
    transfer_func_name = "ANDImpl"
    transfer_func_src = """
    std::vector<APInt> ANDImpl(std::vector<APInt> arg0,
                                     std::vector<APInt> arg1) {
      APInt arg0_0 = arg0[0];
      APInt arg0_1 = arg0[1];
      APInt arg1_0 = arg1[0];
      APInt arg1_1 = arg1[1];
      APInt result_0 = arg0_0 | arg1_0;
      APInt result_1 = arg0_1 & arg1_1;
      return{result_0, result_1};
    }
    """

    names = list(repeat(transfer_func_name, 10))
    srcs = list(repeat(transfer_func_src, 10))
    sound_percent, precise_percent = eval_transfer_func(names, srcs, concrete_op)

    print(f"sound percent:   {sound_percent}")
    print(f"precise percent: {precise_percent}")
'''

# notes
# 1    =  0.849s
# 10   =  0.974s
# 100  =  1.909s
# 250  =  3.719s
# 1000 = 12.361s
