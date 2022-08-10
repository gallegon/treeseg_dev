import numpy as np
from numpy.distutils.core import setup, Extension

treeseg_module = Extension("treeseg",
                    # All .cpp source files to be compiled.
                    sources = [
                        "treeseg.cpp",
                        "Patch.cpp",
                        "disjointpatches.cpp",
                        "pdalfilter.cpp",
                        "Hierarchy.cpp"
                    ],
                    # Directories to any .hpp header files to be included.
                    include_dirs = [
                        np.get_include(),
                        "C:\\Users\\Sam\\Desktop\\projects\\pdalbuilding\\PDAL\\include",
                        "C:\\Users\\Sam\\Desktop\\projects\\pdalbuilding\\PDAL",
                        "libs"
                    ],
                    # -std=c++11
                    language = "c++11",
                    # Directories to any .lib library files.
                    library_dirs = [
                        "C:\\Users\\Sam\\anaconda3\\envs\\treeseg_dev\\Library\\lib"
                    ],
                    # Specific libraries to be linked against (from directories in library_dirs).
                    libraries = [
                        "pdalcpp",
                        "pdal_util"
                    ])


setup(name = "treeseg",
    version = "1.0",
    description = "Tree segmentation in C!",
    ext_modules = [treeseg_module])
