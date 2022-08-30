# Compiles the tree segmentation C++ source code, without the Python extension module.
# Note:
#   -- Must run with the proper conda environment
#   -- Must run with proper MSVC environment variables (recommended to run from MSVC dev command prompt)

import os

source_files = [
    "standalone.cpp",
    "disjointpatches.cpp",
    "disjointtrees.cpp",
    "HDAG.cpp",
    "Hierarchy.cpp",
    "Patch.cpp",
    "pdalfilter.cpp"
]

project_include = [ "include", "libs" ]
library_files = [ "pdalcpp.lib", "pdal_util.lib" ]
compiler_flags = [ "/std:c++14", "/nologo", "/EHsc", "/O2", "/W3" ]

output_folder_exe = os.path.join("build", "standalone")
output_folder_obj = os.path.join("build", "standalone", "obj")
output_file = os.path.join(output_folder_exe, "standalone.exe")

# Fetch path information from environment.
# MSVC and Conda dependencies discovered this way.
msvc_include = os.environ["INCLUDE"].split(";")
msvc_libs = os.environ["LIB"].split(";")

conda_prefix = os.environ["CONDA_PREFIX"]
conda_include = [
    os.path.join(conda_prefix, "include"),
    os.path.join(conda_prefix, "Include"),
    os.path.join(conda_prefix, "Library", "include")
]

conda_libs = [
    conda_prefix,
    os.path.join(conda_prefix, "Library", "lib"),
    os.path.join(conda_prefix, "libs")
]

all_includes = project_include + conda_include + msvc_include
all_libraries = conda_libs + msvc_libs

# Convert all instances of single backslashes into double backslashes,
# while handling originally double backslashes appropriately.
# The palindromic `.replace` ensures that all single backslashe
# become two (needed for proper string escaping within the command string).
def bs2(s):
    return s.replace("\\\\", "\\").replace("\\", "\\\\")

# Create one large string representing the compilation.
cmd = " ".join([
    "cl",
    " ".join(compiler_flags),
    bs2(f'/Fe"{output_file}"'),
    bs2(f'/Fo"{output_folder_obj}\\"'),
    " ".join([bs2(f'/I"{path}"') for path in all_includes]),
    " ".join([os.path.join("src", file) for file in source_files]),
    " ".join(library_files),
    "/link",
    " ".join([bs2(f'/LIBPATH:"{path}"') for path in all_libraries])
])

# Ensure the output folders exist.
os.makedirs(output_folder_exe, exist_ok=True)
os.makedirs(output_folder_obj, exist_ok=True)
# Compile the project.
os.system(cmd)
