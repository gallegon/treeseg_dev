import numpy as np
from numpy.distutils.core import setup, Extension

treeseg_module = Extension('treeseg', sources=['treeseg.cpp'],
                        include_dirs=[
                            np.get_include(),
                            "C:\\Users\\Sam\\Desktop\\pdalbuilding\\PDAL\include",
                            "C:\\Users\\Sam\\Desktop\\pdalbuilding\\PDAL"
                        ],
                        library_dirs=[
                            "C:\\Users\\Sam\\anaconda3\\envs\\treeseg_dev\\Library\\lib"
                        ],
                        libraries=[
                            "pdalcpp",
                            "pdal_util"
                        ])

setup(name = 'treeseg',
        version='1.0',
        description='Tree segmentation in C!',
        ext_modules = [treeseg_module])