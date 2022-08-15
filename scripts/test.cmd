@ECHO OFF

pushd "%~dp0"\..\python

call conda activate treeseg_dev

call python ts_cli.py tests\vernon.json

popd
