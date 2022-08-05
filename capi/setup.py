import numpy as np
from numpy.distutils.core import setup, Extension

treeseg_module = Extension('treeseg', sources=['treeseg.cpp', 'Patch.cpp', "disjointsets.cpp"],
                        include_dirs=[
                            np.get_include(),
                            #"C:\\Users\\Sam\\Desktop\\pdalbuilding\\PDAL\include",
                            #"C:\\Users\\Sam\\Desktop\\pdalbuilding\\PDAL",
                            "libs"
                        ])
# TODO: figure out which libraries to use here.  Ask Mr. Sam Foltz what the
# fuck is going on.  He usually has the answers.
"""
                        ,
                        library_dirs=[
                            "C:\\Users\\Sam\\anaconda3\\envs\\treeseg_dev\\Library\\lib"
                        ],
                        libraries=[
                            "pdalcpp",
                            "pdal_util"
                        ])
"""

setup(name = 'treeseg',
        version='1.0',
        description='Tree segmentation in C!',
        ext_modules = [treeseg_module])
