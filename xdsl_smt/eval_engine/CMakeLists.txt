cmake_minimum_required(VERSION 3.20)
project(EvalEngine LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# Find LLVM and Clang
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set LLVM and Clang components
llvm_map_components_to_libnames(LLVM_LIBS core target x86TargetMCA x86Disassembler x86AsmParser x86CodeGen x86Desc x86Info)
set(CLANG_LIBS clang-cpp)

set(GENERATED_HEADER ${CMAKE_BINARY_DIR}/APInt_bin_string.h)
add_custom_command(
    OUTPUT ${GENERATED_HEADER}
    DEPENDS src/APInt.h
    COMMAND xxd -i ../src/APInt.h > ${GENERATED_HEADER}
    COMMENT "Generating header file from text file"
)

add_executable(eval_engine
  src/main.cpp
  src/jit.h
  src/AbstVal.h
  src/Results.h
  src/Eval.h
  src/APInt.h
  src/warning_suppresor.h
)

target_include_directories(eval_engine PRIVATE
  ${CMAKE_BINARY_DIR}
  ${LLVM_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
)

target_compile_options(eval_engine PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wold-style-cast
    -Wcast-qual
    -Wformat=2
    -O3
    -march=native
    -flto
    -DNDEBUG
    -fstrict-enums
    -Wno-unused-command-line-argument
)

add_custom_target(generate_header DEPENDS ${GENERATED_HEADER})
add_dependencies(eval_engine generate_header)

# Link against LLVM and Clang
target_link_libraries(eval_engine PRIVATE ${LLVM_LIBS} ${CLANG_LIBS})
target_link_options(eval_engine PRIVATE
    -Wl,--copy-dt-needed-entries
)
