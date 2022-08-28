# scripts


## Initial Setup

`env_setup` Setup the required Conda environment and install Python & C++ dependencies.

`env_activate` Activate the treeseg_dev Conda environment.


## Running the Project

`treeseg_test` Run the Python pipeline version of the project.

`server_start` Start the tree segmentation HTTP local server.

`standalone_run` Run the standalone (no Python/Numpy) verion of the project. Currently depends on MSVC for compilation.


## Development Scripts

`treeseg_deploy` Compile and install the C++ Python extension module into the current Conda environment.

`standalone_compile` Compile the standalone (no Python/Numpy) version of the project.

`standalone_run` Run the standalone version of the project.

`clean` Removes compilation and default output files.
