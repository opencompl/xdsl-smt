# xdsl-smt

This repository contains a work-in-progress implementation of an SMTLib dialect for [xDSL](https://github.com/xdslproject/xdsl).

It currently contains the implementation of the core theory of SMTLib and a partial implementation
of the bitvector theory.

It also contains a partial lowering from `pdl`, `arith`, `comb`, and `func` to `smt`, a translation
validation tool between `arith` + `comb` + `func` programs.

## Installation

### Virtual environment

It is recommended to install the project in a virtual environment.
To create a virtual environment, use the following commands:

```bash
# Create a virtual environment
python -m venv venv
# Activate the virtual environment
source venv/bin/activate
```

### Installation

To install the project, use the following commands:

```bash
# Install the project
pip install .
```

### Development installation

To setup an environment for hacking on xdsl-smt, use the following commands:

```bash
# Install the project in editable mode with dev dependencies
pip install -e '.[dev]'
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
or any other solver compatible with SMTLib.

## Running the translation validation tool

The translation validator can be run with
```bash
xdsl-tv file_before.xdsl file_after.xdsl | z3
```

This command will check that the second program is a valid refinement of the first one.

In order to simplify debugging the refinement script, you can pass the `-opt` option
to `xdsl-tv` to simplify obvious expressions and remove the use of the `pair` datatype.

## Verifying PDL rewrites

PDL rewrites can be verified with the `verify-pdl` tool. It takes a single file as input, and will check for the correctness of all rewrites that are present in the file.

It is run with
```
verify-pdl file.mlir
```

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

## Extending the project with new semantics

The lowering to SMT can be extended with new semantics using the fields of the `LowerToSMT` class:
* `type_lowerers` extends the lowering of types to SMT sorts.
* `rewrite_patterns` extends the lowering of operations to SMT operations using rewrite patterns.
* `operation_semantics` extends the lowering of operations to SMT operations using meta-level semantics. Giving semantics this way is necessary to support these operations in `pdl-to-smt`.
* `attribute_semantics` extends the lowering of attributes to SMT attributes using meta-level semantics. Giving semantics this way is necessary to support these attributes in `pdl-to-smt`.

The lowering to SMT from PDL can additionally be extended with the following fields from `PDLToSMT` class:
* `native_rewrites` extends the semantics of functions used in `pdl.apply_native_rewrite`
* `native_constraints` extends the semantics of functions used in `pdl.apply_native_constraint`
* `native_static_constraints` extends the semantics of function used in `pdl.apply_native_constraint`, when the constraint should be checked before the lowering to SMT. This can happen when a PDL rewrite yields an invalid SMT program when this constraint is not satisfied, for instance by creating non-verifying operations.
