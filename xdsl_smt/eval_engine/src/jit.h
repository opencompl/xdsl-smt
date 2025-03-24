#ifndef jit_H
#define jit_H

#include <cstdio>
#include <memory>
#include <string>

#include "APInt_bin_string.h"
#include "warning_suppresor.h"

SUPPRESS_WARNINGS_BEGIN
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/FileManager.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
SUPPRESS_WARNINGS_END

// largely adopted from
// https://blog.memzero.de/llvm-orc-jit/
class Compiler {
private:
  std::unique_ptr<clang::TextDiagnosticPrinter> dp;
  std::unique_ptr<clang::DiagnosticsEngine> de;
  clang::FileSystemOptions fileOpts;
  std::unique_ptr<clang::FileManager> fileManager;

  struct CompileResult {
    std::unique_ptr<llvm::LLVMContext> c;
    std::unique_ptr<llvm::Module> m;
  };

public:
  Compiler() {
    llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> opts =
        new clang::DiagnosticOptions();
    dp = std::make_unique<clang::TextDiagnosticPrinter>(llvm::errs(),
                                                        opts.get());
    de = std::make_unique<clang::DiagnosticsEngine>(nullptr, std::move(opts),
                                                    dp.get(), false);
    fileManager = std::make_unique<clang::FileManager>(fileOpts);
  }

  llvm::Expected<CompileResult> compile(const std::string &code) {
    clang::CompilerInstance clang;
    clang::CompilerInvocation::CreateFromArgs(clang.getInvocation(),
                                              {"<memory>"}, *de);

    clang.getLangOpts().CPlusPlus = true;
    clang.getLangOpts().CPlusPlus11 = true;
    clang.getLangOpts().Bool = true;
    clang.createDiagnostics(fileManager->getVirtualFileSystem(), dp.get(),
                            false);
    clang.getDiagnostics().setIgnoreAllWarnings(true);
    clang.getPreprocessorOpts().addRemappedFile(
        "<memory>", llvm::MemoryBuffer::getMemBuffer(code).release());

    // TODO fiddle with these for perf wins
    clang.getCodeGenOpts().setInlining(
        clang::CodeGenOptions::OnlyAlwaysInlining);
    clang.getCodeGenOpts().OptimizationLevel = 2;

    clang::EmitLLVMOnlyAction action;
    if (!clang.ExecuteAction(action)) {
      return llvm::make_error<llvm::StringError>(
          "Failed to generate LLVM IR from C code!",
          std::make_error_code(std::errc::invalid_argument));
    }

    return CompileResult{
        std::unique_ptr<llvm::LLVMContext>(action.takeLLVMContext()),
        action.takeModule()};
  }
};

std::unique_ptr<llvm::orc::LLJIT> getJit(const std::string &xferSrc) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  std::string apintsrc = std::string(
      reinterpret_cast<const char *>(___src_APInt_h), ___src_APInt_h_len);

  std::string sourceCode = apintsrc + xferSrc;
  auto [context, module] = llvm::cantFail(Compiler().compile(sourceCode));

  std::unique_ptr<llvm::orc::LLJIT> jit =
      llvm::cantFail(llvm::orc::LLJITBuilder().create());

  llvm::orc::ThreadSafeModule mod =
      llvm::orc::ThreadSafeModule(std::move(module), std::move(context));

  llvm::cantFail(jit->addIRModule(std::move(mod)));

  return jit;
}

#endif
