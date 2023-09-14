# xdsl-smt

This repository contains a work-in-progress implementation of an SMTLib dialect for [xDSL](https://github.com/xdslproject/xdsl).

It currently contains the implementation of the core theory of SMTLib and a partial implementation
of the bitvector theory.

It also contains a partial lowering from `pdl`, `arith`, `comb`, and `func` to `smt`, a translation
validation tool between `arith` + `comb` + `func` programs.

## Installation

To install the project, use the following commands:

```bash
# Create a python environment and activate it
python -m venv venv
source ./venv/bin/activate

# Install the requirements
pip install -r requirements.txt -U

# Install the project in editable mode
pip install -e .
```

## Printing SMTLib

When a program only contains `SMTLib` operations and attributes, it can be
printed as an SMTLib script with

```bash
xdsl-smt file.mlir -t smt
```

You can also directly run the SMTLib script with

```bash
xdsl-smt file.mlir -t smt | z3
```
or any other SMTLib solver.

## Running passes with `xdsl-smt`

`xdsl-smt` uses the `-p` command to run passes on a program.
```bash
# Run dce, then convert arith to smt, and output the result in SMTLib form
xdsl-smt file.xdsl -p=dce,lower-to-smt,canonicalize-smt -t smt
```

`xdsl-smt` defines the following passes:
* `dce`: Eliminate dead code.
* `canonicalize-smt`: Apply simple peephole optimizations on SMT programs. This is useful for debugging generated code.
* `lower-pairs`: Try to remove usage of `pair` datatypes. This duplicates function definitions when they return pairs.
* `lower-to-smt`: Lowers `arith`, `comb`, `func` to the `smt` dialect. Can also be extended with additional rewrite
  patterns for new dialects.
* `pdl-to-smt`: Lowers `PDL` rewrites to the `smt` dialect, using the `lower-to-smt` pass. The resulting SMT program
  will check that the rewrite is correct.

## Running the translation validation tool

The translation validator can be run with
```bash
xdsl-tv file_before.xdsl file_after.xdsl | z3
```

This command will check that the second program is a valid refinement of the first one.

In order to simplify debugging the refinement script, you can pass the `-opt` option
to `xdsl-tv` to simplify obvious expressions and remove the use of the `pair` datatype.
