#!/usr/bin/env python3

import argparse
import sys

from xdsl.ir import MLContext, Operation, TypeAttribute, Attribute
from xdsl.parser import Parser

from xdsl.dialects.builtin import Builtin, ModuleOp, IntegerType, IntegerAttr, IntAttr
from xdsl.dialects.func import Func, FuncOp, Return, Call
from xdsl.dialects.arith import Arith, Select
from xdsl.dialects.comb import Comb
from ..dialects.transfer import Transfer, SelectOp, AbstractValueType, TransIntegerType
from ..passes.transfer_inline import FunctionCallInline


def register_all_arguments(arg_parser: argparse.ArgumentParser):
    arg_parser.add_argument("input", type=str, help="path to before input file")

    arg_parser.add_argument(
        "output", type=str, nargs="?", help="path to before output file"
    )

    arg_parser.add_argument("--toMLIR", type=bool, nargs="?", help="transfer to MLIR")

    arg_parser.add_argument("--toDSL", type=bool, nargs="?", help="transfer to DSL")


indent = ""
equals = " = "


def collectWithDelimiter(lst: list[str], delimiter: str) -> str:
    if len(lst)==0:
        return ""
    if len(lst) > 1:
        result = lst[0]
        for ele in lst[1:]:
            result += delimiter + ele
        return result
    return lst[0]


class MLIRtoDSL:
    def __init__(self, module: ModuleOp, transfer_funcs: set[str]) -> None:
        self.module = module
        self.tmp_count = 0
        self.transfer_func = transfer_funcs

    def lowerType(self, type: Attribute) -> str:
        if isinstance(type, AbstractValueType):
            result = "tuple<"
            allTypeStr = [self.lowerType(attr) for attr in type.fields]
            result += collectWithDelimiter(allTypeStr, ", ")
            result += ">"
            return result
        elif isinstance(type, TransIntegerType):
            return "int"
        elif isinstance(type, IntegerType):
            print(str(type))
            return "i" + str(type.width.data)
        else:
            assert False and "unsupported type"

    def lowerFunc(self, func: FuncOp) -> str:
        global indent
        result = indent
        # lower func signature

        argsNameAndType = [
            arg.name_hint + " : " + self.lowerType(arg.type) for arg in func.args
        ]
        argsStr = collectWithDelimiter(argsNameAndType, ", ")
        # We assume one func has only one return value
        assert len(func.get_return_op().operands) == 1
        returnType = self.lowerType(func.get_return_op().operands[0].type)
        funcName = func.sym_name.data
        result += "def " + funcName + "(" + argsStr + ") -> " + returnType + " : \n"

        indent += "\t"
        self.tmp_count = 0
        for op in func.body.ops:
            result += self.lowerOp(op) + "\n"

        indent = indent[:-1]
        result += "\n"
        # might need extra lines after function body

        return result

    def lowerAttributes(self, attrs: dict[str, Attribute]) -> str:
        attrLst: list[str] = []
        for k, v in attrs.items():
            if isinstance(v, IntegerAttr):
                attrLst.append(k + " = " + str(v.value.data))
            else:
                print(k, v)
                assert False and "Don't support other attrs right now"
        return collectWithDelimiter(attrLst, ", ")

    def lowerOp(self, op: Operation) -> str:
        result = indent

        operandsName = [operand.name_hint for operand in op.operands]
        operandsStr = collectWithDelimiter(operandsName, ", ")
        if len(op.attributes) >= 1:
            attrStr = self.lowerAttributes(op.attributes)
            if len(operandsName) >= 1:
                operandsStr += ", " + attrStr

        for ele in operandsName:
            assert ele is not None
        # return oper has no returned value
        if isinstance(op, Return):
            return result + "return " + operandsStr

        if op.results[0].name_hint is None:
            op.results[0].name_hint = "tmp_" + str(self.tmp_count)
            self.tmp_count += 1
        returnedName = op.results[0].name_hint
        returnedType = self.lowerType(op.results[0].type)
        operBody = ""
        if isinstance(op, Call):
            operBody = str(op.callee)[1:] + "(" + operandsStr + ")"
        else:
            operBody = op.name + " " + operandsStr

        return result + returnedName + " : " + returnedType + " = " + operBody

    def lowerModule(self) -> str:
        global indent
        result = indent
        for op in self.module.body.ops:
            if isinstance(op, FuncOp) and op.sym_name.data in self.transfer_func:
                result += self.lowerFunc(op)
                result += "\n"
        return result


def splitAndStrip(s: str, delimiter: str) -> list[str]:
    tokens = [token.strip() for token in s.split(delimiter)]
    tokens = list(filter(lambda token: len(token) != 0, tokens))
    return tokens


class DSLType:
    def __init__(self, ty: str) -> None:
        self.type = ty
        self.typeStr: dict[str, str] = {}

    @staticmethod
    def parseType(s: str):
        return DSLType(s)

    def toStr(self) -> str:
        if self.type in self.typeStr:
            return self.typeStr[self.type]
        # achieve by replacement
        mlirTy = self.type
        # index -> index
        # int -> !transfer.integer
        # tuple -> !transfer.abs_value
        # < -> <[
        # > -> ]>
        mlirTy=mlirTy.replace("tuple", "!transfer.abs_value")
        mlirTy=mlirTy.replace("int", "!transfer.integer")
        mlirTy=mlirTy.replace("<", "<[")
        mlirTy=mlirTy.replace(">", "]>")
        self.typeStr[self.type] = mlirTy
        return mlirTy


class DSLValue:
    def __init__(self, name: str, ty: DSLType) -> None:
        self.name = name
        self.type = ty

    @staticmethod
    def parseValue(s: str):
        assert ":" in s
        tokens = splitAndStrip(s, ":")
        assert len(tokens) == 2
        return DSLValue(tokens[0], DSLType.parseType(tokens[1]))

    def getName(self) -> str:
        return "%" + self.name

    def getType(self) -> str:
        return self.type.toStr()

    def toStr(self) -> str:
        return self.getName() + " : " + self.getType()


class DSLAttribute:
    def __init__(self, name: str, val: str, ty: DSLType) -> None:
        self.val = val
        self.name = name
        self.type = ty

    @staticmethod
    def parseAttribute(s: str):
        # we only support index type for now
        assert "=" in s
        tokens = splitAndStrip(s, "=")
        assert len(tokens) == 2
        return DSLAttribute(tokens[0], tokens[1], DSLType("index"))

    def toStr(self) -> str:
        return self.name + " = " + self.val + " : " + self.type.toStr()


# a:tuple<int,int>, b:int should be a:tuple<int,int> and b:int
# Split string considering brackets
def splitToken(s: str, delimiter: str) -> list[str]:
    assert len(delimiter) == 1
    result: list[str] = []
    lastIdx = -1
    bracketCnt = 0
    for idx, c in enumerate(s):
        if c == "<":
            bracketCnt += 1
        elif c == ">":
            bracketCnt -= 1
        elif c == delimiter and bracketCnt == 0:
            result.append(s[lastIdx + 1 : idx].strip())
            lastIdx = idx
        else:
            pass
    result.append(s[lastIdx + 1 :].strip())
    return result


def generateOperationType(args: list[DSLValue], results: list[DSLValue]) -> str:
    argsType: list[str] = [arg.getType() for arg in args]
    argsTypeStr = collectWithDelimiter(argsType, ",")

    resultsType: list[str] = [res.getType() for res in results]
    resultsTypeStr = collectWithDelimiter(resultsType, ",")
    if len(resultsType) != 1:
        resultsTypeStr = "(" + resultsTypeStr + ")"

    return "(" + argsTypeStr + ") -> " + resultsTypeStr


class DSLOperation:
    def __init__(
        self,
        opName: str,
        args: list[DSLValue],
        attrs: list[DSLAttribute],
        results: list[DSLValue],
    ) -> None:
        self.opName = opName
        self.args = args
        self.attrs = attrs
        self.results = results

    @staticmethod
    def parseOperation(s: str, valueMap: dict[str, DSLValue]):
        # special handling on return
        if s.startswith("return"):
            tokens=splitAndStrip(s, " ")
            args:list[DSLValue]=[]
            for token in tokens[1:]:
                assert token in valueMap
                args.append(valueMap[token])
            return DSLOperation("return",args,[],[])


        # a : tuple<int,int> = foo b, c, index=5
        firstEq = s.find("=")
        # a : tuple<int,int>
        resultStr = s[:firstEq].strip()
        resultTokens = splitToken(resultStr, ",")
        results: list[DSLValue] = [DSLValue.parseValue(token) for token in resultTokens]

        # foo b, c, index=5
        rightPart = s[firstEq + 1 :].strip()
        firstSpace = rightPart.find(" ")
        # foo
        opName = rightPart[:firstSpace]
        # b, c, index=5
        argsStr = rightPart[firstSpace + 1 :]
        argsToken = splitAndStrip(argsStr, ",")
        attrs: list[DSLAttribute] = []
        args: list[DSLValue] = []
        for arg in argsToken:
            if "=" in arg:
                attrs.append(DSLAttribute.parseAttribute(arg))
            else:
                assert arg in valueMap
                args.append(valueMap[arg])
        return DSLOperation(opName, args, attrs, results)

    def toStr(self) -> str:
        argsName: list[str] = [arg.getName() for arg in self.args]    
        argsStr = collectWithDelimiter(argsName, ",")
        typeStr = generateOperationType(self.args, self.results)

        if self.opName == "return":
            return indent + '"func.return"(' + argsStr + ") : " + typeStr
    
        resultsName: list[str] = [res.getName() for res in self.results]
        resultsStr = collectWithDelimiter(resultsName, ",")
        attrs: list[str] = [attr.toStr() for attr in self.attrs]
        attrsStr = ""
        if len(attrs) > 0:
            attrsStr = "{" + collectWithDelimiter(attrs, ",") + "}"
        return (
            indent
            + resultsStr
            + ' = "'
            + self.opName
            + '"('
            + argsStr
            + ")"
            + attrsStr
            + " : "
            + typeStr
        )


FUNC_DEF_KEYWORD = "def"


class DSLFuncntion:
    def __init__(
        self, funcName: str, args: list[DSLValue], operations: list[DSLOperation]
    ) -> None:
        self.operations = operations
        self.funcName = funcName
        self.args = args

    @staticmethod
    def parseFunction(lines: list[str]):
        assert len(lines) >= 2
        assert lines[0].startswith(FUNC_DEF_KEYWORD)
        # parse func signature and set value map
        args: list[DSLValue] = []
        # lines[0]:
        # def ADDImpl(arg0 : tuple<int, int>, arg1 : tuple<int, int>) -> tuple<int, int> :
        funcSignature = lines[0][len(FUNC_DEF_KEYWORD) :].strip()
        firstBracket = funcSignature.find("(")
        lastBracket = funcSignature.find(")")

        funcName = funcSignature[:firstBracket].strip()

        valueMap: dict[str, DSLValue] = {}
        argsStr = funcSignature[firstBracket + 1 : lastBracket]
        argsTokens = splitToken(argsStr, ",")
        for token in argsTokens:
            args.append(DSLValue.parseValue(token))
            valueMap[args[-1].name] = args[-1]

        operations: list[DSLOperation] = []
        for line in lines[1:]:            
            operations.append(DSLOperation.parseOperation(line, valueMap))
            for res in operations[-1].results:
                valueMap[res.name] = res
        return DSLFuncntion(funcName, args, operations)

    def toStr(self) -> str:
        global indent
        functionHead = indent + '"func.func"() ({\n'
        argsStr: list[str] = [arg.toStr() for arg in self.args]

        blockHead = indent + "^bb0({args}):\n".format(
            args=collectWithDelimiter(argsStr, ",")
        )

        indent += "\t"
        body = ""
        for op in self.operations:
            body += op.toStr()
            body += "\n"
        indent = indent[:-1]

        returnVal = self.operations[-1].args
        assert self.operations[-1].opName == "return"
        funcType = generateOperationType(self.args, returnVal)
        # need handle return
        functionEnd = '}}) {{function_type ={func_ty}, sym_name = "{func_name}" }} : () -> ()'.format(
            func_name=self.funcName, func_ty=funcType
        )
        return functionHead + blockHead + body + functionEnd


class DSLModule:
    def __init__(self, functions: list[DSLFuncntion]) -> None:
        self.functions = functions

    @staticmethod
    def parseModule(lines: list[str]):
        lastIdx = 0
        isInFunc = False
        functions: list[DSLFuncntion] = []        
        for idx, line in enumerate(lines):            
            if line.startswith(FUNC_DEF_KEYWORD):
                lastIdx = idx
                isInFunc = True
            elif line == "" and isInFunc:
                functions.append(DSLFuncntion.parseFunction(lines[lastIdx:idx]))
                isInFunc = False
        return DSLModule(functions)

    def toStr(self) -> str:
        moduleHead = '"builtin.module"() ({\n'
        global indent
        indent += "\t"

        body = ""
        for func in self.functions:
            body += func.toStr()
            body += "\n"

        indent = indent[:-1]
        # maybe need add attr "builtin.NEED_VERIFY"=[["MUL","MULImpl"]]
        moduleEnd = "}): () -> ()\n"

        return moduleHead + body + moduleEnd


def parseDSLModule(lines: list[str]) -> DSLModule:
    return DSLModule.parseModule(lines)


def main():
    ctx = MLContext()
    arg_parser = argparse.ArgumentParser()
    register_all_arguments(arg_parser)
    args = arg_parser.parse_args()

    # Register all dialects
    ctx.load_dialect(Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(Comb)
    ctx.load_dialect(Transfer)

    # Parse the files
    def parse_mlir(file: str | None) -> Operation:
        if file is None:
            f = sys.stdin
        else:
            f = open(file)

        parser = Parser(ctx, f.read())
        module = parser.parse_module()
        return module

    module = None
    if args.toDSL:
        module = parse_mlir(args.input)
        assert isinstance(module, ModuleOp)
        transfer_func: set[str] = set()
        for func_pair in module.attributes["builtin.NEED_VERIFY"]:
            concrete_funcname, transfer_funcname = func_pair
            transfer_func.add(transfer_funcname.data)

        func_name_to_func: dict[str, FuncOp] = {}
        for func in module.ops:
            if isinstance(func, FuncOp):
                func_name_to_func[func.sym_name.data] = func
        FunctionCallInline(False, func_name_to_func).apply(ctx, module)

        parser = MLIRtoDSL(module, transfer_func)
        if args.output:
            print(args.output)
            with open(args.output) as out:
                out.write(parser.lowerModule())
        else:
            print(parser.lowerModule())
    elif args.toMLIR:
        with open(args.input) as input:
            lines = input.readlines()
            for i in range(len(lines)):
                lines[i]=lines[i].strip()
            dslModule = parseDSLModule(lines)            
            if args.output:
                with open(args.output) as out:
                    out.write(dslModule.toStr())
            else:
                print(dslModule.toStr())
    else:
        assert "Unknown type keyword, should to 'DLS' or 'MLIR'" and False
