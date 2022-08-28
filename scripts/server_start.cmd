@echo off

setlocal

pushd "%~dp0"\..\treesegvis

call python ts_http.py %1

popd

endlocal
