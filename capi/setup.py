import numpy as np
from numpy.distutils.core import setup, Extension

import os
import platform

pdal_include_path = os.path.join(os.environ["CONDA_PREFIX"], "Library", "include")
pdal_library_path = os.path.join(os.environ["CONDA_PREFIX"], "Library", "lib")



if platform.system() == "Windows":
    extra_compile_args = []
else:
    extra_compile_args = ["-ggdb", "-fno-inline-functions", "-O0"]

treeseg_ext_module = Extension("treeseg_ext",
                    # All .cpp source files to be compiled (in the src directory).
                    sources = [ os.path.join("src", s) for s in [
                        "treeseg_ext.cpp",
                        "Patch.cpp",
                        "disjointpatches.cpp",
                        "pdalfilter.cpp",
                        "Hierarchy.cpp",
                        "HDAG.cpp",
                        "disjointtrees.cpp"
                    ]],
                    # Directories to any .hpp header files to be included.
                    include_dirs = [
                        np.get_include(),
                        pdal_include_path,
                        "include",
                        "libs"
                    ],
                    # -std=c++11
                    language = "c++11",
                    # Directories to any .lib library files.
                    library_dirs = [
                        pdal_library_path,
                    ],
                    # Specific libraries to be linked against (from directories in library_dirs).
                    libraries = [
                        "pdalcpp",
                        "pdal_util"
                    ],
                    extra_compile_args = extra_compile_args)


setup(name = "treeseg_ext",
    version = "1.0",
    description = "Tree segmentation in C!",
    ext_modules = [ treeseg_ext_module ])
