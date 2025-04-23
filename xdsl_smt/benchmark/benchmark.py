from xdsl_smt.cli.synth_transfer import run
from os import mkdir, path, listdir, remove
from math import nan
from multiprocessing import Pool
from random import randint
from json import dump

from xdsl_smt.eval_engine.eval import AbstractDomain

# vals that xuanyu sugessted
# NUM_PROGS = 100
# NUM_ITERS = 30
# NUM_ROUNDS = 2000
NUM_PROGS = 75
NUM_ITERS = 25
NUM_ROUNDS = 1250
COND_LEN = 10
SOL_SIZE = 0
NUM_ABD_P = 25
BWIDTH = 4
WEIGHT_DSL = True
PROGRAM_LENGTH = 40

# something faster
# PROGRAM_LENGTH = 40
# NUM_PROGS = 50
# NUM_ITERS = 25
# NUM_ROUNDS = 25
# COND_LEN = 15
# SOL_SIZE = 0
# NUM_ABD_P = 10
# BWIDTH = 4
# WEIGHT_DSL = True


def rm_r(dir: str):
    try:
        files = listdir(dir)
        for file in files:
            file_path = path.join(dir, file)
            if path.isfile(file_path):
                remove(file_path)
    except OSError:
        print(f"Error occurred while deleting files in {dir}")


def setup_outputs(domain: str, func: str) -> str:
    try:
        mkdir("outputs")
    except FileExistsError:
        pass

    output_folder = path.join("outputs", f"{domain}_{func}")
    try:
        mkdir(output_folder)
    except FileExistsError:
        rm_r(output_folder)

    return output_folder


def synth_run(args: tuple[str, str, str, int]) -> dict[str, float | str]:
    func_name = args[0]
    domain = args[1]
    fname = args[2]
    seed = args[3]

    try:
        output_folder = setup_outputs(domain, func_name)
        res = run(
            AbstractDomain[domain],
            num_programs=NUM_PROGS,
            num_iters=NUM_ITERS,
            total_rounds=NUM_ROUNDS,
            condition_length=COND_LEN,
            solution_size=SOL_SIZE,
            num_abd_procs=NUM_ABD_P,
            bitwidth=BWIDTH,
            weighted_dsl=WEIGHT_DSL,
            program_length=PROGRAM_LENGTH,
            random_seed=seed,
            transfer_functions=fname,
            outputs_folder=output_folder,
        )

        return {
            "Domain": domain,
            "Function": func_name,
            "Sound Proportion": res.get_sound_prop() * 100,
            "Exact Proportion": res.get_exact_prop() * 100,
            "Seed": seed,
            "Notes": "",
        }
    except Exception as e:
        return {
            "Domain": domain,
            "Function": func_name,
            "Sound Proportion": nan,
            "Exact Proportion": nan,
            "Seed": seed,
            "Notes": f"Run was terminated: {e}",
        }


def main() -> None:
    seed = randint(1, 1_000_000)

    start_dir = path.join("tests", "synth")
    xfer_funcs = {
        ("KnownBits", "Add"): "knownBitsAdd.mlir",
        ("KnownBits", "And"): "knownBitsAnd.mlir",
        ("KnownBits", "Ashr"): "knownBitsAshr.mlir",
        ("KnownBits", "Lshr"): "knownBitsLshr.mlir",
        ("KnownBits", "Mods"): "knownBitsMods.mlir",
        ("KnownBits", "Modu"): "knownBitsModu.mlir",
        ("KnownBits", "Mul"): "knownBitsMul.mlir",
        ("KnownBits", "Or"): "knownBitsOr.mlir",
        ("KnownBits", "Sdiv"): "knownBitsSdiv.mlir",
        ("KnownBits", "Shl"): "knownBitsShl.mlir",
        ("KnownBits", "Udiv"): "knownBitsUdiv.mlir",
        ("KnownBits", "Xor"): "knownBitsXor.mlir",
        ("ConstantRange", "Add"): "integerRangeAdd.mlir",
        ("ConstantRange", "And"): "integerRangeAnd.mlir",
        ("ConstantRange", "Ashr"): "integerRangeAshr.mlir",
        ("ConstantRange", "Lshr"): "integerRangeLshr.mlir",
        ("ConstantRange", "Mods"): "integerRangeMods.mlir",
        ("ConstantRange", "Modu"): "integerRangeModu.mlir",
        ("ConstantRange", "Mul"): "integerRangeMul.mlir",
        ("ConstantRange", "Or"): "integerRangeOr.mlir",
        ("ConstantRange", "Sdiv"): "integerRangeSdiv.mlir",
        ("ConstantRange", "Shl"): "integerRangeShl.mlir",
        ("ConstantRange", "Udiv"): "integerRangeUdiv.mlir",
        ("ConstantRange", "Xor"): "integerRangeXor.mlir",
    }

    def get_path(x: str) -> str:
        return "integerRange" if x == "ConstantRange" else "knownBits"

    xfer_funcs = {
        k: path.join(start_dir, get_path(k[0]), v) for k, v in xfer_funcs.items()
    }

    inputs = [
        (func_name, domain_name, xfer_func_fname, seed)
        for (domain_name, func_name), xfer_func_fname in xfer_funcs.items()
    ]

    with Pool() as p:
        data = p.map(synth_run, inputs)

    with open("data.json", "w") as f:
        dump(data, f, indent=2)


if __name__ == "__main__":
    main()
