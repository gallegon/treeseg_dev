@ECHO OFF

setlocal

pushd "%~dp0"\..\treesegvis

@REM call python -m http.server --directory src 8080
call python ts_http.py %1

popd

endlocal