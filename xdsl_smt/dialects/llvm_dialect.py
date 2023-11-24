# pyright: reportUnusedImport=false

# We import all the LLVM operations here, so we only access them from here.
# Once we will have added all LLVM operations in xds, we can remove this file.
from xdsl.dialects.llvm import (
    ExtractValueOp,
    InsertValueOp,
    UndefOp,
    AllocaOp,
    GEPOp,
    IntToPtrOp,
    NullOp,
    LoadOp,
    StoreOp,
    GlobalOp,
    AddressOfOp,
    FuncOp,
    CallOp,
    ReturnOp,
    ConstantOp,
    CallIntrinsicOp,
    LLVMStructType,
    LLVMPointerType,
    LLVMArrayType,
    LLVMVoidType,
    LLVMFunctionType,
    LinkageAttr,
    CallingConventionAttr,
    FastMathAttr,
)
from xdsl.dialects import llvm
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition

LLVM: Dialect = llvm.LLVM
