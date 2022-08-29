@echo off

set ENV_NAME=treeseg_dev

call conda create -n %ENV_NAME%

call conda env update -n %ENV_NAME% -f ..\environment.yaml

call conda activate %ENV_NAME%

call python -m pip install pdal
