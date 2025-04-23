#!/bin/bash

start_setup=$(date +%s)
# get system dependencies
sudo apt update
sudo apt install -y cmake ninja-build lld python3.10-venv

mkdir experiment/
cd experiment/

# get binutils
git clone --depth 1 git://sourceware.org/git/binutils-gdb.git binutils

# get and build llvm
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 87adafcd2e248fa69d1f776a9e60f95df03b885d
mkdir build
cd build
# TODO the path on this command depends on the user running the experiment
cmake -GNinja -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_EH=ON -DBUILD_SHARED_LIBS=ON \
-DLLVM_BINUTILS_INCDIR=/users/dkennedy/experiment/binutils/include/ \
-DCMAKE_BUILD_TYPE=Release -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_ASSERTIONS=ON \
-DLLVM_USE_LINKER=lld \
-DLLVM_ENABLE_PROJECTS="llvm;clang;mlir" ../llvm
ninja
cd ../../

# bulid eval engine
git clone https://github.com/Hatsunespica/xdsl-smt.git
cd xdsl-smt/xdsl_smt/eval_engine/
git checkout benchmark
mkdir build && cd build
# TODO this path too
cmake .. -D  CMAKE_CXX_COMPILER=/users/dkennedy/experiment/llvm-project/build/bin/clang++ \
-D CMAKE_PREFIX_PATH=/users/dkennedy/experiment/llvm-project/build
make
cd ../../../

# setup python env
python -m venv venv
source venv/bin/activate
pip install .
pip install -e '.[dev]'
end_setup=$(date +%s)

echo "export SETUP_TIME=$((end_setup-start_setup))" >> ~/.bashrc

# run
benchmark-synth
