# capi

This is the Python C++ extension for the Tree Segmentation project.

The source code `.cpp` files are located in `src/`, the `.hpp` files in `include/`.

`libs` contains any library headers for compilation. Currently only the C++ boost library is needed.
The other library dependencies are handled by `setup.py` with Conda. As such the proper Conda environment is required.

`treeseg.cpp` is the Python extension module which acts as the interface from our C++ code to/from Python.
