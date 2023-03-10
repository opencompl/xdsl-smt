# xdsl-smt

This repository containts a work in progress implementation of an SMTLib dialect for xDSL.

It currently contains the implementation of the core theory of SMTLib, and a partial implementation
of the bitvector theory.

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

When a program only contain `SMTLib` operations and attributes, it can be 
printed as a SMTLib script with

```bash
python xdsl-smt.py file.xdsl -t smt
```

## Running passes with `xdsl-smt`

`xdsl-smt` uses the `-p` command to run passes on a program.
```bash
# Run dce, then convert arith to smt, and output the result in SMTLib form
python xdsl-smt.py file.xdsl -p=dce,arith_to_smt -t smt
```

`xdsl-smt` defines the following passes:
* `dce`: Eliminate dead code.
* `canonicalize_smt`: Apply simple peephole optimizations on SMT programs. This is useful for debugging generated code.
* `lower_pairs`: Try to remove usage of `pair` datatypes. This duplicate function definitions when they return pairs.
* `arith_to_smt`: Convert `arith` operations and attributes to the `smt` dialect
