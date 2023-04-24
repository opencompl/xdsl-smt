# xdsl-smt

This repository contains a work-in-progress implementation of an SMTLib dialect for xDSL.

It currently contains the implementation of the core theory of SMTLib and a partial implementation
of the bitvector theory.

It also contains a partial lowering from `pdl`, `arith`, and `func` to `smt`, a translation
validation tool between `arith` + `func` programs.

## Installation

To install the project for developers, follow the following commands:

```bash
# Create a python environment and activate it
python -m venv venv
source ./venv/bin/activate

# Install the requirements
pip install -r requirements.txt -U

# Install the project in editable mode
pip install -e .
```

## Using `xdsl-smt` with xDSL or MLIR

`xdsl-smt` can parse and print xDSL or MLIR programs with
```bash
# Parse and print an xDSL program
python xdsl-smt.py file.xdsl
# Parse an MLIR program and prints it as MLIR
python xdsl-smt.py file.mlir -t mlir

# Parse an MLIR propram and saves it to an xDSL file
python xdsl-smt.py file.mlir -o new.xdsl
```

## Printing SMTLib

When a program only contains `SMTLib` operations and attributes, it can be
printed as a SMTLib script with

```bash
python xdsl-smt.py file.xdsl -t smt
```

You can also directly run the SMTLib script with

```bash
python xdsl-smt.py file.xdsl -t smt | z3
```
or any other SMTLib solver.

## Running passes with `xdsl-smt`

`xdsl-smt` uses the `-p` command to run passes on a program.
```bash
# Run dce, then convert arith to smt, and output the result in SMTLib form
python xdsl-smt.py file.xdsl -p=dce,arith-to-smt -t smt
```

`xdsl-smt` defines the following passes:
* `dce`: Eliminate dead code.
* `canonicalize_smt`: Apply simple peephole optimizations on SMT programs. This is useful for debugging generated code.
* `lower-pairs`: Try to remove usage of `pair` datatypes. This duplicates function definitions when they return pairs.
* `arith-to-smt`: Convert `arith` operations and attributes to the `smt` dialect
* `pdl-to-smt`: Convert `PDL` rewrites on `arith` operations to the `smt` dialect,
   which can be directly ran with SMT-Lib to check for correctness of the rewrite.

## Running the translation validation tool

The translation validator can be run with
```bash
./xdsl-tv.py file_before.xdsl file_after.xdsl | z3
```

This command will check that the second program is a valid refinement of the first one.

In order to simplify debugging the refinement script, you can pass the `-opt` option
to `xdsl-tv.py` to simplify obvious expressions and remove the use of the `pair` datatype.
