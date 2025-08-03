import subprocess as sp
import time
from typing import Sequence, Iterable

from xdsl.dialects.func import FuncOp
from xdsl.dialects.pdl import PatternOp

MLIR_ENUMERATE = "./mlir-fuzz/build/bin/mlir-enumerate"
EXCLUDE_SUBPATTERNS_FILE = f"/tmp/exclude-subpatterns-{time.time()}.mlir"
BUILDING_BLOCKS_FILE = f"/tmp/building-blocks-{time.time()}.mlir"


def _read_program_from_enumerator(enumerator: sp.Popen[str]) -> str | None:
    """Read a single program from the enumerator."""
    program_lines = list[str]()
    assert enumerator.stdout is not None
    while True:
        output = enumerator.stdout.readline()

        # End of program marker
        if output == "// -----\n":
            return "".join(program_lines)

        # End of file
        if not output:
            return None

        # Add the line to the program lines otherwise
        program_lines.append(output)


def enumerate_programs(
    max_num_args: int,
    num_ops: int,
    bv_widths: str,
    building_blocks: list[list[FuncOp]] | None,
    illegals: list[PatternOp],
    dialect_path: str,
    additional_options: Sequence[str] = (),
) -> Iterable[str]:
    """Enumerate all programs up to a given size."""
    if building_blocks is not None:
        with open(BUILDING_BLOCKS_FILE, "w") as f:
            for blocks in building_blocks:
                for program in blocks:
                    f.write(str(program))
                    f.write("\n// -----\n")
                f.write("// +++++\n")

    with open(EXCLUDE_SUBPATTERNS_FILE, "w") as f:
        for illegal in illegals:
            f.write(str(illegal))
            f.write("\n// -----\n")

    enumerator = sp.Popen(
        [
            MLIR_ENUMERATE,
            dialect_path,
            "--configuration=smt",
            f"--smt-bitvector-widths={bv_widths}",
            # Make sure CSE is applied.
            "--cse",
            # Prevent any non-deterministic behavior (hopefully).
            "--seed=1",
            f"--max-num-args={max_num_args}",
            f"--max-num-ops={num_ops}",
            "--pause-between-programs",
            "--mlir-print-op-generic",
            f"--building-blocks={BUILDING_BLOCKS_FILE if building_blocks is not None else ''}",
            f"--exclude-subpatterns={EXCLUDE_SUBPATTERNS_FILE}",
            *additional_options,
        ],
        text=True,
        stdin=sp.PIPE,
        stdout=sp.PIPE,
    )

    while (source := _read_program_from_enumerator(enumerator)) is not None:
        # Send a character to the enumerator to continue.
        assert enumerator.stdin is not None
        enumerator.stdin.write("a")
        enumerator.stdin.flush()

        yield source
