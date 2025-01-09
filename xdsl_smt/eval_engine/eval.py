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


def make_xfer_header(concrete_op_expr: str) -> str:
    return (
        """
    #include <llvm/ADT/APInt.h>
    #include <llvm/Support/KnownBits.h>
    #include <tuple>
    using llvm::APInt;

    uint8_t concrete_op(const uint8_t a, const uint8_t b) {
        return %s;
    }
    """
        % concrete_op_expr
    )


def make_xfer_wrapper(func_name: str) -> str:
    return (
        """
    llvm::KnownBits synth_function_wrapper(const llvm::KnownBits &lhs,
                                           const llvm::KnownBits &rhs) {
      const auto [res_zero, res_one] =
          %s({lhs.Zero, lhs.One}, {rhs.Zero, rhs.One});

      llvm::KnownBits res;

      res.Zero = res_zero;
      res.One = res_one;

      return res;
    }
    """
        % func_name
    )


def eval_transfer_func(
    xfer_names: list[str],
    xfer_srcs: list[str],
    concrete_op_expr: str,
) -> list[tuple[float, float]]:
    transfer_func_header = make_xfer_header(concrete_op_expr)

    xfer_func_wrappers = "\n".join([make_xfer_wrapper(x) for x in xfer_names])
    all_xfer_src = "\n".join(xfer_srcs)

    base_dir = path.join("xdsl_smt", "eval_engine")
    cur_dir = os.getcwd()
    synth_code_path = path.join(cur_dir, base_dir, "src", "synth.cpp")

    with open(synth_code_path, "w") as f:
        f.write(f"{transfer_func_header}\n{all_xfer_src}\n{xfer_func_wrappers}")

    try:
        os.mkdir(path.join(cur_dir, base_dir, "build"))
    except FileExistsError:
        pass

    os.chdir(path.join(base_dir, "build"))

    run(get_build_cmd(), stdout=PIPE)
    eval_output = run(["./EvalEngine"], stdout=PIPE)

    def get_float(s: str) -> float:
        return float(s.split(":")[1])

    os.chdir(cur_dir)

    eval_output_lines = eval_output.stdout.decode("utf-8").split("\n")
    sound_percent = get_float(eval_output_lines[5])
    precise_percent = get_float(eval_output_lines[6])

    return sound_percent, precise_percent


if __name__ == "__main__":
    concrete_op_expr = "a & b"
    transfer_func_name = "ANDImpl"
    transfer_func_src = """
    std::tuple<APInt, APInt> ANDImpl(std::tuple<APInt, APInt> arg0,
                                     std::tuple<APInt, APInt> arg1) {
      APInt arg0_0 = std::get<0>(arg0);
      APInt arg0_1 = std::get<1>(arg0);
      APInt arg1_0 = std::get<0>(arg1);
      APInt arg1_1 = std::get<1>(arg1);
      APInt result_0 = arg0_0 | arg1_0;
      APInt result_1 = arg0_1 & arg1_1;
      std::tuple<APInt, APInt> result = std::make_tuple(result_0, result_1);
      return result;
    }
    """

    sound_percent, precise_percent = eval_transfer_func(
        [transfer_func_name], [transfer_func_src], concrete_op_expr
    )

    print(f"sound percent:   {sound_percent}")
    print(f"precise percent: {precise_percent}")
