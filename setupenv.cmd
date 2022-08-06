set ENV_NAME=treeseg_dev

call conda create -n %ENV_NAME%

call conda env update -n %ENV_NAME% -f environment.

pip install pdal
