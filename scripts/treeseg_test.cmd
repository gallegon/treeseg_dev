@echo off

setlocal

pushd "%~dp0"\..\python

call conda activate treeseg_dev

call python ts_cli.py ..\tests\pipeline_saveall.json

popd

endlocal
