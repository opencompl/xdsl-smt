# pyright: reportWildcardImportFromLibrary=false
# pyright: reportConstantRedefinition=false

# We import all the LLVM operations here, so we only access them from here.
# Once we will have added all LLVM operations in xds, we can remove this file.
from xdsl.dialects.llvm import *
from xdsl.dialects import llvm
from xdsl.ir import Dialect

LLVM: Dialect = llvm.LLVM
