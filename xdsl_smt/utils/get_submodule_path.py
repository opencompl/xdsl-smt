import os


def get_mlir_fuzz_path() -> str:
    """Get the path to the mlir-fuzz submodule."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "..", "..", "mlir-fuzz")


def get_mlir_fuzz_executable_path(executable_name: str) -> str:
    """Get the path to an executable in the mlir-fuzz submodule."""
    return os.path.join(get_mlir_fuzz_path(), "build", "bin", executable_name)
