from typing import List, Optional

from pathlib import Path
import numpy
from setuptools import setup
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext


if __name__ == "__main__":
    workspace_dir = Path("")
    output_binding_file = workspace_dir / "src" / "py_nn" / "binding.cpp"
    output_stub_file = workspace_dir / "src" / "py_nn" / "py_nn.pyi"
    hpp_files = list(workspace_dir.rglob("*.hpp"))
    cpp_files = list(workspace_dir.rglob("*.cpp"))

    # name
    name = "nn"

    # If include dir is None just set as empty list
    include_dirs = [
        "/home/wsl/Projects_code/vcpkg/installed/x64-linux/include/", 
        "/usr/include/",
        "/home/wsl/Projects_code/xtensor-python/include",
        numpy.get_include()
    ]

    # Make the module
    ext_modules = [
        Pybind11Extension(
            name,
            cpp_files,
            include_dirs=[pybind11.get_include(), *include_dirs],
            extra_compile_args = ["-lopenblas", "-DPYBIND11_DETAILED_ERROR_MESSAGES"],
            extra_link_args = None,
            libraries=['blas'])
    ]

    # Finally exectute the setup
    setup(
        name=name,
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
    )