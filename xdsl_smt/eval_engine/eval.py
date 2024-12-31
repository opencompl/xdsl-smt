import os
from subprocess import run, PIPE


def eval_transfer_func(
    transfer_func_name: str, transfer_func_src: str, concrete_op_expr: str
) -> tuple[float, float]:
    transfer_func_headers = (
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

    transfer_func_wrapper = (
        """
    llvm::KnownBits synth_function_wrapper(const llvm::KnownBits &lhs,
                                           const llvm::KnownBits &rhs) {
      const auto res_vec =
          %s({lhs.Zero, lhs.One}, {rhs.Zero, rhs.One});

      llvm::KnownBits res;

      res.Zero = res_vec[0];
      res.One = res_vec[1];

      return res;
    }
    """
        % transfer_func_name
    )
    base_dir="xdsl_smt/eval_engine/"
    cur_dir=os.getcwd()+"/"
    with open(cur_dir+base_dir+"src/synth.cpp", "w") as f:
        f.write(
            f"{transfer_func_headers}\n{transfer_func_src}\n{transfer_func_wrapper}"
        )

    '''
    try:
        os.mkdir(base_dir+"build")
    except FileExistsError:
        pass
    '''

    os.chdir(base_dir+"build")

    run(["cmake", ".."], stdout=PIPE)
    run(["make"], stdout=PIPE)
    eval_output = run(["./EvalEngine"], stdout=PIPE)

    def get_float(s: str) -> float:
        return float(s.split(":")[1])

    os.chdir(cur_dir)

    eval_output_lines = eval_output.stdout.decode("utf-8").split("\n")
    sound_percent = get_float(eval_output_lines[5])
    precise_percent = get_float(eval_output_lines[6])

    return sound_percent, precise_percent


'''
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
        transfer_func_name, transfer_func_src, concrete_op_expr
    )

    print(f"sound percent:   {sound_percent}")
    print(f"precise percent: {precise_percent}")
'''